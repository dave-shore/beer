import stanza
import os
from string import punctuation
import re
from joblib import Parallel, delayed
from itertools import product
from typing import List, Sequence, Union, Callable
from math import exp
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    from ..src.mbeer.utils import _find_tokens, batch_generator
except ImportError:
    import sys
    from pathlib import Path

    src_path = Path(__file__).parent.parent / "src" / "mbeer"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from utils import _find_tokens, batch_generator

WordSpec = Union[int, str, Sequence[Union[int, str]]]


def semantic_distances(sentences, phrases_a, phrases_b):

    assert len(sentences) == len(phrases_a) == len(phrases_b)
    model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0")
    mask_token = model.tokenizer.mask_token

    tokenized_sentences = model.tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True
    ).encodings

    # Get tokens positions
    tokens_positions_a = [
        _find_tokens(sent, phrase_a, encoding)
        for sent, phrase_a, encoding in zip(sentences, phrases_a, tokenized_sentences)
    ]
    tokens_positions_b = [
        _find_tokens(sent, phrase_b, encoding)
        for sent, phrase_b, encoding in zip(sentences, phrases_b, tokenized_sentences)
    ]

    # We have to put in as many mask tokens as the number of tokens in the phrase to keep the same length of the sentence
    sentences_a = [
        sent.replace(phrase_b, " ".join([mask_token] * sum(p[1] - p[0] for p in pos_b)))
        for sent, phrase_b, pos_b in zip(sentences, phrases_b, tokens_positions_b)
    ]
    sentences_b = [
        sent.replace(phrase_a, " ".join([mask_token] * sum(p[1] - p[0] for p in pos_a)))
        for sent, phrase_a, pos_a in zip(sentences, phrases_a, tokens_positions_a)
    ]

    embeddings_a = model.encode(
        sentences_a,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
        output_value="token_embeddings",
        batch_size=16,
    )
    embeddings_b = model.encode(
        sentences_b,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
        output_value="token_embeddings",
        batch_size=16,
    )

    embeddings_a = [
        embedding[tokens_position[0][0] : tokens_position[0][1]]
        for embedding, tokens_position in zip(embeddings_a, tokens_positions_a)
    ]
    embeddings_b = [
        embedding[tokens_position[0][0] : tokens_position[0][1]]
        for embedding, tokens_position in zip(embeddings_b, tokens_positions_b)
    ]

    distance_matrices = [
        cdist(embedding_a.cpu().numpy(), embedding_b.cpu().numpy())
        for embedding_a, embedding_b in zip(embeddings_a, embeddings_b)
    ]
    lsas = [
        linear_sum_assignment(distance_matrix) for distance_matrix in distance_matrices
    ]
    distances = [
        distance_matrix[row_ind, col_ind].sum()
        for distance_matrix, (row_ind, col_ind) in zip(distance_matrices, lsas)
    ]

    return distances


def _extract_dependency_tree(sentence) -> dict:
    """Build a lightweight tree dict from a Stanza Sentence."""
    return {
        "text": sentence.text,
        "words": {w.id: w.text for w in sentence.words},
        "heads": {w.id: w.head for w in sentence.words},
        "deprel": {w.id: w.deprel for w in sentence.words},
    }


def parse_dependency_trees(
    sentences: List[str],
    lang: str = "en",
    nlp: stanza.Pipeline | None = None,
    processors: str = "tokenize,mwt,pos,lemma,depparse",
    batch_size: int = 64,
    num_workers: int | None = None,
) -> List[dict]:
    """Bulk-parse sentences with Stanza and return one dependency tree per input sentence.

    Each tree is a dict with keys ``text``, ``words`` (id -> token), ``heads`` (id -> head id,
    0 = root), and ``deprel`` (id -> relation label). Word ids follow Stanza's 1-based indexing.
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() // 2, batch_size // 2)

    if nlp is None:
        nlp = stanza.Pipeline(
            lang,
            processors=processors,
            tokenize_no_ssplit=True,
            download_method=None,
        )

    trees = []
    for batch in tqdm(
        batch_generator(sentences, batch_size),
        total=(len(sentences) + batch_size - 1) // batch_size,
        desc="Dependency tree parsing",
    ):
        concatenated_batch = "\n\n".join([sent.replace("\n", " ") for sent in batch])
        doc = nlp.process(concatenated_batch)
        if len(doc.sentences) == 0:
            raise ValueError(f"Stanza produced no sentences for input: {doc.text!r}")

        doc_trees = Parallel(n_jobs=num_workers)(
            delayed(_extract_dependency_tree)(sent) for sent in doc.sentences
        )
        trees.extend(doc_trees)

    return trees


def _match_consecutive_tokens(tree: dict, tokens: Sequence[str]) -> List[int]:
    """Resolve a multi-token string expression to consecutive word ids."""
    ids_sorted = sorted(tree["words"])
    tokens_lower = [t.lower() for t in tokens]

    for start in range(len(ids_sorted) - len(tokens) + 1):
        window = [
            tree["words"][ids_sorted[start + j]].lower() for j in range(len(tokens))
        ]
        if window == tokens_lower:
            return ids_sorted[start : start + len(tokens)]

    # Fallback: union of individual token matches (non-consecutive MWE).
    matched = []
    not_matched = []
    for tok in tokens_lower:
        hits = [
            wid
            for wid, text in tree["words"].items()
            if text.lower() == tok
            or text.lower().strip(punctuation) == tok
            or text.strip("b").lower() == tok.strip("'").lower()
        ]
        if not hits:
            not_matched.append(tok)
            continue
        matched.extend(hits)

    if not matched:
        if any(re.search(r"[^\w]", tok) for tok in tokens):
            try:
                tokens = sum([re.split(r"[^\w]", tok) for tok in tokens], [])
                return _match_consecutive_tokens(tree, tokens)
            except ValueError:
                pass
        else:
            raise ValueError(
                f"Token {not_matched!r} not found in tree for {tree['text']!r}"
            )

    return sorted(set(matched))


def _resolve_word_ids(tree: dict, word_spec: WordSpec) -> List[int]:
    """Map a word or multi-word expression to Stanza word ids in ``tree``."""
    if isinstance(word_spec, int):
        if word_spec not in tree["words"]:
            raise ValueError(f"Word id {word_spec} not in tree for {tree['text']!r}")
        return [word_spec]

    if isinstance(word_spec, str):
        word_spec = word_spec.strip(punctuation)
        matches = [
            wid
            for wid, text in tree["words"].items()
            if text.lower() == word_spec.lower()
            or text.lower().strip(punctuation) == word_spec.lower()
            or text.strip("b").lower() == word_spec.strip("'").lower()
        ]
        if not matches:
            raise ValueError(
                f"Word {word_spec!r} not found in tree for {tree['text']!r}"
            )
        return matches

    if isinstance(word_spec, (list, tuple)):
        if len(word_spec) == 0:
            raise ValueError("Empty word expression")
        if isinstance(word_spec[0], int):
            missing = [wid for wid in word_spec if wid not in tree["words"]]
            if missing:
                raise ValueError(f"Word ids {missing} not in tree for {tree['text']!r}")
            return list(word_spec)
        return _match_consecutive_tokens(tree, word_spec)

    raise TypeError(f"Unsupported word spec type: {type(word_spec)}")


def _dependency_tree_distance(tree: dict, word_id_a: int, word_id_b: int) -> int:
    """Shortest-path length (in edges) between two words in an undirected dependency tree."""
    if word_id_a == word_id_b:
        return 0

    heads = tree["heads"]
    # Be sure they are all integers
    heads = {int(k): int(v) for k, v in heads.items()}
    word_id_a = int(word_id_a)
    word_id_b = int(word_id_b)

    ancestors_a = {}
    depth = 0
    node = word_id_a
    while True:
        ancestors_a[node] = depth
        parent = heads[node]
        if parent == 0:
            break
        node = parent
        depth += 1

    node = word_id_b
    depth = 0
    while node not in ancestors_a:
        parent = heads[node]
        if parent == 0:
            return ancestors_a[word_id_a] + depth
        node = parent
        depth += 1

    return ancestors_a[node] + depth


def syntactic_distance(
    tree: dict, words_a: WordSpec, words_b: WordSpec, tokenizer: Callable
) -> float:
    """Minimum syntactic distance (number of dependency edges) between two word expressions.

    ``words_a`` and ``words_b`` may be a single Stanza word id, a token string, or a
    sequence of ids/strings for a multi-word expression. For expressions with multiple
    words, the distance is the minimum over all cross-pairs of resolved word ids.
    """
    if isinstance(words_a, str) and re.search(r"[^\w]", words_a) is not None:
        words_a = tokenizer(words_a)
    if isinstance(words_b, str) and re.search(r"[^\w]", words_b) is not None:
        words_b = tokenizer(words_b)

    ids_a = _resolve_word_ids(tree, words_a)
    ids_b = _resolve_word_ids(tree, words_b)

    return float(
        min(_dependency_tree_distance(tree, a, b) for a, b in product(ids_a, ids_b))
    )


def concatenate_tokens(tokens, tokenizer_special_char):
    if tokenizer_special_char == "##":
        tokens = [
            token.replace("##", "") if token.startswith("##") else " " + token
            for token in tokens
        ]
    elif tokenizer_special_char == "Ġ":
        tokens = [token.replace("Ġ", " ") for token in tokens]
    return "".join(tokens)


def jaccard_similarity(
    L1, L2, beta=0.5, add_ngrams=0, length_penalty=0, str_diff=False
):

    if str_diff:
        assert isinstance(L1, str) and isinstance(L2, str)
        sim = SequenceMatcher(None, L1, L2).ratio()
        return sim

    # L1 is the true entity, L2 is the predicted entity
    if add_ngrams > 1:
        L1 = L1 + [
            " ".join(L1[i : i + add_ngrams]) for i in range(len(L1) - add_ngrams + 1)
        ]
        L2 = L2 + [
            " ".join(L2[i : i + add_ngrams]) for i in range(len(L2) - add_ngrams + 1)
        ]

    # when true entity is short, add a penalty to the error
    penalty = 1 - exp(-length_penalty * len(L1))

    return (
        len(set(L1) & set(L2))
        * penalty
        / (len(set(L1)) * beta + len(set(L2)) * (1 - beta))
    )


def matching_str_pairs(a1, a2, b1, b2, **kwargs):

    J11 = jaccard_similarity(a1, b1, **kwargs)
    J12 = jaccard_similarity(a1, b2, **kwargs)
    J21 = jaccard_similarity(a2, b1, **kwargs)
    J22 = jaccard_similarity(a2, b2, **kwargs)

    M = np.matrix([[J11, J22], [J21, J12]])

    return M.mean(axis=1).max()
