"""Main script"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import gc
import itertools
from itertools import chain, combinations
import random
import json
import math
import os
import re
from typing import List, Tuple, Dict, Set
import warnings

from datasets import load_dataset
from huggingface_hub import HfApi
import pandas as pd
from scipy.stats import chi2
from numpy import cumsum
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)

hf_api = HfApi()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EPS = torch.finfo(torch.float32).eps
_WORD_RE = re.compile(r"\w+")

# Handle both relative and absolute imports
try:
    from .pullback import (
        OutputOnlyModel,
        compute_pred_ids_and_eq_class_emb_ids,
        pullback,
    )
    from .utils import batch_generator, timing_decorator, _find_tokens
    from .baseline_functions import (
        mean_integrated_directional_gradients,
        shapley_interaction_quantification,
    )
    from .mbrd import RelationDisambiguation
except ImportError:
    # If relative import fails, try absolute import
    import sys
    from pathlib import Path

    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from pullback import (
        OutputOnlyModel,
        compute_pred_ids_and_eq_class_emb_ids,
        pullback,
    )
    from utils import batch_generator, timing_decorator, _find_tokens
    from baseline_functions import (
        mean_integrated_directional_gradients,
        shapley_interaction_quantification,
    )
    from mbrd import RelationDisambiguation


def _resolve_torch_device(d) -> torch.device:
    """Canonical CUDA device for dict keys and set_device (cuda -> cuda:0)."""

    x = (
        torch.device(d)
        if d is not None
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if isinstance(x, str):
        x = torch.device(x, 0)
    elif isinstance(x, torch.device):
        if getattr(x, "type", None) == "cuda" and x.index is None:
            x = torch.device("cuda", 0)
        else:
            pass
    else:
        raise ValueError(f"Invalid device type: {type(x)}")

    return x


def fill_shape(shape: Tuple[int, ...], force_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(s if s in shape else -1 for s in force_shape)


def _mask_spans(
    encoding: torch.Tensor, spans: List[Tuple[int, int]], mask_id: int
) -> torch.Tensor:
    """Clone `encoding` and overwrite all positions in any of `spans` with `mask_id`."""
    if isinstance(spans[0][0], tuple):
        spans = list(chain.from_iterable(spans))
    out = encoding.clone()
    for s in spans:
        out[s[0] : s[1]] = mask_id
    return out


def pad_traces_and_eigenvectors(L: List[torch.Tensor]) -> torch.Tensor:

    # Avoid 0-dim tensors
    L = [T.unsqueeze(0) if T.dim() == 0 else T for T in L]

    shapes = [t.shape for t in L]
    dims = {len(s) for s in shapes}
    dim = max(dims)
    # Should be 2 for predictions, 3 for traces and 4 for eigenvectors
    if len(dims) > 1:
        try:
            force_shape = next(s for s in shapes if len(s) == dim)
            L = [T.reshape(fill_shape(T.shape, force_shape)) for T in L]
        except RuntimeError:
            raise ValueError(
                "All tensors must have the same number of dimensions or be reshapeable to the same number of dimensions"
            )

    if dim < 3:
        max_last_dim = max(T.shape[-1] for T in L)
        padded_tensors = [
            T.unsqueeze(-1)
            if T.dim() == 1
            else torch.nn.functional.pad(T, (0, max_last_dim - T.shape[1]))
            for T in L
        ]
        return torch.nn.utils.rnn.pad_sequence(
            padded_tensors, batch_first=True, padding_value=0
        )

    padding_shape = tuple(max(along_dim) for along_dim in zip(*shapes))

    for i, T in enumerate(L):
        if i % 3:
            # For pairs (j,i), with i < j, we pad the left side of the output dim (dim 1) and the right side of the input dim (dim 2)
            left_padding = tuple(
                chain.from_iterable(
                    [
                        [padding_shape[j] - T.shape[j], 0]
                        if j % 2
                        else [0, padding_shape[j] - T.shape[j]]
                        for j in range(len(T.shape))
                    ][::-1]
                )
            )
            T = torch.nn.functional.pad(T, left_padding).to(
                dtype=torch.get_default_dtype()
            )
        else:
            # For pairs (j,i), with i > j, we pad the right side of the output dim (dim 1) and the left side of the input dim (dim 2)
            right_padding = tuple(
                chain.from_iterable(
                    [
                        [0, padding_shape[j] - T.shape[j]]
                        if j % 2
                        else [padding_shape[j] - T.shape[j], 0]
                        for j in range(len(T.shape))
                    ][::-1]
                )
            )
            T = torch.nn.functional.pad(T, right_padding).to(
                dtype=torch.get_default_dtype()
            )

        # For tensors that combine two entities it doesn't matter, we treat them as left-padded
        L[i] = T

    output = torch.cat(L)

    if torch.isnan(output).any():
        warnings.warn("NaN values in output")

    return output


def build_pca_head(model: AutoModel, reduced_dim: int = None):
    try:
        from .new_layers import PCAHead, PCAHeadModel
    except ImportError:
        import sys
        from pathlib import Path

        src_path = Path(__file__).parent.parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from new_layers import PCAHead, PCAHeadModel

    if reduced_dim is None:
        reduced_dim = model.config.hidden_size // 4

    base_model = model.base_model if hasattr(model, "base_model") else model
    pca_head = PCAHead(model.config.hidden_size, reduced_dim, model.device, model.dtype)

    return PCAHeadModel(base_model, pca_head)


def train_lm_head(
    model_name: str,
    device: torch.device,
    mask_id: int,
    n_epochs: int = 10,
    inputs: torch.Tensor = None,
    attn_mask: torch.Tensor = None,
    encoder_name: str = "encoder",
    patience: int = None,
    tol: float = 1e-6,
    hf_token: str = None,
):

    parent_model_info = hf_api.model_info(model_name)
    parent_model_name = parent_model_info.card_data.get("base_model", None)
    if parent_model_name is None and "-finetuned" in model_name:
        parent_model_name = re.sub("-finetuned.*$", "", model_name)
        if not hf_api.repo_exists(parent_model_name):
            parent_model_name = None
    if isinstance(parent_model_name, list):
        parent_model_name = parent_model_name[0]
    ignore_modules = ["dropout", "activation", "act_fn"]

    if patience is None:
        patience = n_epochs // 10

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    masked_inputs = torch.where(
        torch.rand_like(inputs.to(dtype=torch.get_default_dtype())) < 0.15,
        mask_id,
        inputs,
    )

    if parent_model_name is not None:
        print(f"Loading parent model {parent_model_name}", flush=True)
        try:
            output_only_model = OutputOnlyModel(
                AutoModelForMaskedLM.from_pretrained(
                    parent_model_name.strip("-.]["),
                    trust_remote_code=True,
                    token=hf_token,
                ).to(device, dtype=torch.get_default_dtype()),
                encoder_name=encoder_name,
            )
        except ValueError as e:
            print(f"Error loading parent model {parent_model_name}: {e}", flush=True)
            print("Trying with LLM2Vec", flush=True)

            from peft import PeftModel

            new_name = (
                "McGill-NLP/LLM2Vec-"
                + parent_model_name.replace("/", "-")
                .replace(".", "")
                .replace("meta-llama", "meta")
                .title()
                + "-Instruct-mntp"
            )
            config = AutoConfig.from_pretrained(
                new_name, trust_remote_code=True, token=hf_token
            )
            inner_model = AutoModel.from_pretrained(
                new_name,
                trust_remote_code=True,
                token=hf_token,
                config=config,
                attn_implementation="flash_attention_2",
            )
            plain_model = PeftModel.from_pretrained(inner_model, new_name)
            output_only_model = OutputOnlyModel(
                plain_model.to(device, dtype=torch.get_default_dtype()),
                encoder_name=encoder_name,
            )

    else:
        print(
            f"Couldn't find parent model for {model_name}, initializing MLM head from scratch",
            flush=True,
        )
        output_only_model = OutputOnlyModel(
            AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True, token=hf_token
            ).to(device, dtype=torch.get_default_dtype()),
            encoder_name=encoder_name,
        )
        are_weights_initialized = {
            k: all(torch.all(p.diff() < 1e-6) for p in v.parameters())
            for k, v in output_only_model.model.named_modules()
            if hasattr(v, "parameters") and all(ign not in k for ign in ignore_modules)
        }
        uninitialized_modules = [k for k, v in are_weights_initialized.items() if v]
        uninitialized_parameters = [
            p
            for v in uninitialized_modules
            for p in output_only_model.model.get_submodule(v).parameters()
        ]

        mask_token_id = (
            tokenizer.mask_token_id
            if tokenizer.mask_token_id is not None
            else tokenizer.pad_token_id
        )

        if len(uninitialized_modules) > 0:
            modules_path = os.path.join("models", model_name.split("/")[-1])

            if os.path.exists(modules_path) and len(os.listdir(modules_path)) > 0:
                for k in uninitialized_modules:
                    try:
                        output_only_model.model.get_submodule(k).load_state_dict(
                            torch.load(os.path.join(modules_path, f"{k}.pt"))
                        )
                    except FileNotFoundError:
                        warnings.warn(
                            f"Could not load weights for module {k} from {modules_path}"
                        )

                print(
                    f"Loaded weights for {len(uninitialized_modules)} modules",
                    flush=True,
                )

                return output_only_model

            os.makedirs(
                os.path.join("models", model_name.split("/")[-1]), exist_ok=True
            )

            output_only_model.model.requires_grad_(False)
            for p in uninitialized_parameters:
                p.requires_grad_(True)

            with torch.enable_grad():
                for module in uninitialized_modules:
                    output_only_model.model.get_submodule(module).train()

                loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
                optimizer = torch.optim.AdamW(
                    uninitialized_parameters, lr=1e-3, weight_decay=1e-4
                )

                dataset = TensorDataset(masked_inputs, inputs, attn_mask)
                dataloader = DataLoader(
                    dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2
                )
                loss_sequence = torch.ones(patience, device=device) * float("inf")

                for epoch in trange(n_epochs, desc="Training LM head"):
                    epoch_loss = []
                    for minibatch in dataloader:
                        selected_inputs, selected_targets, selected_attn_mask = (
                            minibatch
                        )
                        selected_inputs = selected_inputs.to(device)
                        selected_targets = selected_targets.to(device)
                        selected_attn_mask = selected_attn_mask.to(device)
                        y = output_only_model.model(
                            selected_inputs, attention_mask=selected_attn_mask
                        ).logits
                        masked_tokens_mask = torch.repeat_interleave(
                            selected_inputs.reshape(selected_inputs.shape[0], -1, 1)
                            == mask_token_id,
                            y.shape[-1],
                            dim=-1,
                        )
                        masked_logits = y[masked_tokens_mask].reshape(-1, y.shape[-1])
                        masked_targets = selected_targets[
                            masked_tokens_mask[..., 0]
                        ].reshape(-1)
                        optimizer.zero_grad()
                        vocab_size = masked_logits.shape[-1]
                        loss = loss_fn(
                            masked_logits.view(-1, vocab_size),
                            F.one_hot(masked_targets.view(-1), vocab_size).to(
                                masked_logits.device, dtype=masked_logits.dtype
                            ),
                        )
                        loss.backward()
                        current_loss = optimizer.step()
                        epoch_loss.append(current_loss.item())

                    epoch_loss = torch.as_tensor(epoch_loss).mean().item()
                    loss_sequence[epoch % patience] = epoch_loss
                    if torch.all(loss_sequence <= max(epoch_loss, tol)):
                        break
                    print(f"Epoch {epoch}, loss: {epoch_loss:.4f}")

            optimizer.zero_grad(set_to_none=True)
            del optimizer, loss_fn

            output_only_model.model.requires_grad_(False)

            for module in uninitialized_modules:
                output_only_model.model.get_submodule(module).eval()
                torch.save(
                    output_only_model.model.get_submodule(module).state_dict(),
                    os.path.join("models", model_name.split("/")[-1], f"{module}.pt"),
                )

            print(f"Trained {len(uninitialized_modules)} modules", flush=True)

    return output_only_model


def get_entity_spans_from_predictions(
    predictions: torch.Tensor,
    pred_to_mask: Dict[int, int],
    row_lens: torch.Tensor | None = None,
    batch_tokens_per_row: List[List[str]] | None = None,
    delete_tokens: Set[str] | None = None,
) -> List[List[Tuple[int, int]]]:

    # Get the entity spans as (start, end_exclusive) tuples.
    # pred_mask values: 0 = outside/padding, 1 = B/U (entity start), 2 = I/L (inside).
    if len(pred_to_mask) > 2:
        # Vectorised lookup avoids the per-element Python dispatch of `Tensor.apply_`.
        max_label_id = max(pred_to_mask.keys())
        lookup = torch.zeros(max_label_id + 1, dtype=torch.int8)
        for k, v in pred_to_mask.items():
            lookup[k] = v
        pred_mask = lookup[predictions.long()]
    else:
        # Binary fallback: re-encode runs of "1" so consecutive entity tokens are
        # merged into a single span by the BIO-aware boundary detector below
        # (1 = first token, 2 = continuation), preserving the original behaviour.
        binary = (predictions > 0).to(torch.int8)
        prev_binary = torch.nn.functional.pad(binary[:, :-1], (1, 0), value=0)
        pred_mask = binary + binary * prev_binary
    prev_mask = torch.nn.functional.pad(pred_mask[:, :-1], (1, 0), value=0)

    positions = torch.arange(pred_mask.shape[1], device=pred_mask.device).repeat(
        pred_mask.shape[0], 1
    )
    # An entity starts at any B/U, or at an I/L that follows an O (defensive against malformed BIO).
    is_start = (pred_mask == 1) | ((pred_mask == 2) & (prev_mask == 0))
    is_start = is_start & (positions < row_lens.unsqueeze(1).to(positions.device))
    # An entity ends (exclusive) where the previous token was inside an entity and the current token is O or starts a new entity.
    is_end = (prev_mask != 0) & ((pred_mask == 0) | (pred_mask == 1))
    is_end = is_end & (positions < row_lens.unsqueeze(1).to(positions.device))

    ent_spans = []
    for i in range(pred_mask.shape[0]):
        starts = is_start[i].nonzero(as_tuple=True)[0].tolist()
        ends = is_end[i].nonzero(as_tuple=True)[0].tolist()
        # Close any entity that runs up to the last valid token (no following O to mark its end).
        if len(starts) == len(ends) + 1 and row_lens is not None:
            ends.append(row_lens[i].item())
        elif len(starts) > len(ends) + 1:
            ends = ends[: len(starts)]
        ent_spans.append(list(zip(starts, ends)))

    if delete_tokens is not None and batch_tokens_per_row is not None:
        ent_spans = [
            [
                span
                for span in span_list
                if span[0] < span[1]
                and span[1] <= row_lens[i].item()
                and all(
                    tok not in delete_tokens and _WORD_RE.search(tok) is not None
                    for tok in batch_tokens_per_row[i][span[0] : span[1]]
                )
            ]
            for i, span_list in enumerate(ent_spans)
        ]

    return ent_spans


def construct_negative_samples(
    expanded_batch: torch.Tensor,
    eq_class_emb_ids: torch.Tensor,
    vocab_size: int,
    fraction: float = 0.5,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Build a negative-sample copy of ``expanded_batch``.

    For every row, sample ``fraction`` of that row's valid (non-padding)
    ``eq_class_emb_ids`` positions and overwrite the tokens at those positions
    in the copied tensor with random ids drawn uniformly from
    ``[0, vocab_size - 1]``. The original ``expanded_batch`` is left untouched.
    """

    negatives = torch.empty_like(expanded_batch).copy_(expanded_batch)
    seq_len = negatives.shape[-1]

    for i, row in enumerate(eq_class_emb_ids):
        valid_positions = row[torch.logical_and(row >= 0, row < seq_len)].unique()
        if valid_positions.numel() == 0:
            continue

        num_to_sample = max(1, int(round(fraction * valid_positions.numel())))
        perm = torch.randperm(valid_positions.numel(), generator=generator)
        chosen_positions = valid_positions[perm[:num_to_sample]]

        random_tokens = torch.randint(
            0,
            vocab_size,
            (chosen_positions.numel(),),
            dtype=negatives.dtype,
            generator=generator,
        )
        negatives[i, chosen_positions] = random_tokens

    return negatives


@torch.no_grad()
@timing_decorator
def forward_backward_pass(
    batch: List[str],
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    mlm_model: AutoModelForMaskedLM,
    gold_spans: List[List[Tuple[int, int]]] = None,
    min_num_eigenvectors: int = 32,
    max_num_eigenvectors: int = None,
    max_tokens_per_entity: int = 5,
    max_retained_vocab: int = 50,
    minibatch_size: int = 3,
    encoder_name: str = "encoder",
    insert_negative_samples: bool = False,
):

    max_len = (
        model.config.max_position_embeddings - 2
        if hasattr(model.config, "max_position_embeddings")
        else model.config.n_ctx - 2
    )
    pred_to_mask = (
        {
            k: int(v.startswith(("B", "U")))
            + 2 * int(v.startswith(("I", "L")) and not v.endswith("0"))
            for k, v in model.config.id2label.items()
        }
        if hasattr(model.config, "id2label")
        else {}
    )
    local_device = model.device
    if local_device.type == "cuda":
        torch.cuda.set_device(local_device)

    encoded_inputs = tokenizer(
        batch,
        padding="longest",
        add_special_tokens=False,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    ).to(local_device)
    delete_tokens = set(tokenizer.special_tokens_map.values())
    batch_tokens_per_row = [enc.tokens for enc in encoded_inputs.encodings]
    attn_mask = encoded_inputs.attention_mask
    row_lens = attn_mask.sum(dim=1).to(dtype=torch.int32)
    num_labels = len(pred_to_mask) if pred_to_mask else 2
    # Threshold from power of Chi-Squared test for uniformity of distribution
    probit_threshold = 1 / num_labels + math.sqrt(
        chi2.isf(0.95, num_labels - 1) / (2 * num_labels)
    )
    probit_threshold = min(probit_threshold, 0.5)

    logit_threshold = torch.nn.Sequential(
        torch.nn.Softmax(dim=-1), torch.nn.Threshold(probit_threshold, 0)
    )

    raw_predictions = model(**encoded_inputs).logits
    predictions = logit_threshold(
        raw_predictions * attn_mask.unsqueeze(-1).to(raw_predictions.device)
    ).cpu()
    if predictions.shape[-1] > 1:
        predictions = predictions.argmax(dim=-1)
    else:
        predictions = predictions.round().to(dtype=torch.int8).squeeze(-1)
    # predictions.shape = (batch_size, max_len)

    ent_spans = get_entity_spans_from_predictions(
        predictions, pred_to_mask, row_lens, batch_tokens_per_row, delete_tokens
    )

    # If gold spans are provided, we need to remove any found spans that overlap with the gold spans and insert the gold spans in place of the found spans
    if gold_spans is not None:
        new_spans = []
        for found_span_list, gold_span_list in zip(ent_spans, gold_spans):
            if len(gold_span_list) == 1:
                gold_span_list.pop(0)

            filtered_span_list = [
                found
                for found in sorted(set(found_span_list))
                if not any(
                    min(found[1], gold[1]) - max(found[0], gold[0]) > 0
                    for gold in gold_span_list
                )
                and found[1] > found[0]
            ]
            if len(filtered_span_list) > 1:
                new_spans.append(filtered_span_list)
            else:
                new_spans.append([])

        ent_spans = new_spans

    else:
        gold_spans = [[] for _ in ent_spans]

    # Eliminate documents with no valid span combinations
    valid_documents = [
        i
        for i, ent_list in enumerate(ent_spans)
        if len(ent_list) > 1 or len(gold_spans[i]) > 1
    ]
    if len(valid_documents) == 0:
        warnings.warn("No valid span combinations found")
        return {}

    batch = [
        batch[i] for i in valid_documents
    ]  # batch_size = number of valid documents
    selected_inputs = encoded_inputs.input_ids[
        valid_documents
    ]  # shape = (batch_size, max_len_documents), values are int64 in [0, vocab_size]
    attn_mask = attn_mask[
        valid_documents
    ]  # shape = (batch_size, max_len_documents), values are bool
    row_lens = row_lens[
        valid_documents
    ]  # shape = (batch_size,), values are int32 in [0, max_len_documents]
    predictions = predictions[
        valid_documents
    ]  # shape = (batch_size, max_len_documents), values are int8 in [0, num_labels]
    raw_predictions = raw_predictions[
        valid_documents
    ]  # shape = (batch_size, max_len_documents, num_labels), values are float32
    ent_spans = [
        sorted(set(ent_spans[i] + gold_spans[i])) for i in valid_documents
    ]  # shape = (batch_size, N_entities), values are tuples (start, end)
    span_combinations = [
        list(combinations(ent_list, 2)) for ent_list in ent_spans
    ]  # shape = (batch_size, N_combinations(len(spans), 2)), values are tuples ((start_i, end_i), (start_j, end_j))

    first_order = [
        ent_span_list + span_combo_list
        for ent_span_list, span_combo_list in zip(ent_spans, span_combinations)
    ]

    # Expand the batch so that for each input sentence there are N_entities + N_pairs inputs: one with each entity masked, and one with both entities masked
    # The order of everything is, for each document: e_1, e_2, ..., e_N, (e_1, e_2), (e_1, e_3), ..., (e_N-1, e_N)
    mask_tok_id = (
        tokenizer.mask_token_id
        if tokenizer.mask_token_id is not None
        else tokenizer.pad_token_id
    )
    selected_inputs = selected_inputs.to("cpu")
    expanded_batch = [
        [_mask_spans(encoding, [s], mask_tok_id) for s in ent_and_combo_list]
        for encoding, ent_and_combo_list in zip(selected_inputs, first_order)
    ]
    expanded_batch = torch.nn.utils.rnn.pad_sequence(
        list(chain.from_iterable(expanded_batch)),
        batch_first=True,
    )
    # expanded_batch.shape = (sum(N_entities_per_document + N_pairs_per_document), max_len)
    # The final order of processing will be: entity_s, entity_o, pair
    # Re-ordering index for later. Precompute per-document offsets and
    # item-to-index maps so each (s, o, (s,o)) lookup is O(1) instead of
    # the original O(N) list-prefix sum + O(N) list.index per pair.
    new_order: List[int] = []
    offset = 0
    for doc_tuples in first_order:
        for i, tup in enumerate(doc_tuples):
            if isinstance(tup[0], tuple):
                s, o = tup
                new_order.append(offset + doc_tuples.index(s))
                new_order.append(offset + doc_tuples.index(o))
                new_order.append(offset + i)
            else:
                pass
        offset += len(doc_tuples)

    if expanded_batch.shape[0] == 0 or len(new_order) == 0:
        raise ValueError("No valid span combinations found")

    # Cache sampled positions per unique entity span so identical spans
    # (the common case across pair combinations within a document) share
    # the same tensor instead of rebuilding it.
    pred_ids, eq_class_emb_ids = compute_pred_ids_and_eq_class_emb_ids(
        span_combinations, ent_spans, max_tokens_per_entity
    )
    assert pred_ids.numel() > 0, "No valid masked entity spans found"
    assert eq_class_emb_ids.numel() > 0, "No valid entity spans to perturb found"
    # pred_ids.shape = (sum(N_entities_per_document + N_pairs_per_document), max_masked_entity_spans_length)
    # eq_class_emb_ids.shape = (sum(N_entities_per_document + N_pairs_per_document), max_perturbed_entity_spans_length)

    # Define separators on the predictions to get the correct entity spans:
    # - for each entity_span, the separators are the ordered lengths of the others in the same document
    # - for each span_combo, the separators are the lengths of the two entities in the combo
    independent_entity_separators = list(
        chain(
            *[
                [
                    tuple(
                        min(max_tokens_per_entity, other[1] - other[0])
                        for other in ent_span_list
                        if other != ent
                    )
                    for ent in ent_span_list
                ]
                + [
                    (
                        min(max_tokens_per_entity, s[1] - s[0]),
                        min(max_tokens_per_entity, o[1] - o[0]),
                    )
                    for s, o in span_combo_list
                ]
                for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)
            ]
        )
    )

    dependent_entity_separators = list(
        chain(
            *[
                [
                    (min(max_tokens_per_entity, ent[1] - ent[0]),)
                    for ent in ent_span_list
                ]
                + [
                    (
                        min(max_tokens_per_entity, s[1] - s[0]),
                        min(max_tokens_per_entity, o[1] - o[0]),
                    )
                    for s, o in span_combo_list
                ]
                for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)
            ]
        )
    )

    # Select the vocabulary in order to avoid carrying the whole vocabulary into Jacobians
    repetitions = torch.as_tensor([len(doc_tuples) for doc_tuples in first_order])
    repeated_attn_mask = attn_mask.repeat_interleave(
        repetitions.to(attn_mask.device), dim=0
    )

    del model
    if isinstance(mlm_model, OutputOnlyModel):
        output_only_model = mlm_model.to(local_device)
    else:
        output_only_model = OutputOnlyModel(
            mlm_model.to(local_device), encoder_name=encoder_name
        )

    # Firdt prediction to select the vocabulary
    selected_inputs = torch.where(
        selected_inputs < output_only_model.model.config.vocab_size,
        selected_inputs,
        output_only_model.model.config.vocab_size - 1,
    )
    token_predictions = output_only_model.model(
        selected_inputs.to(local_device)
    ).logits.cpu()
    selected_vocab = token_predictions.topk(
        k=max_retained_vocab, dim=-1
    ).indices.repeat_interleave(repetitions, dim=0)

    # Reorder into (entity_s, entity_o, pair) triplets so that each
    # minibatch of 3 fed to pullback forms a valid grouping.
    # The new order says in which position each entity span (first) and pair (then) are in the expanded batch
    triplet_order = torch.as_tensor(new_order)
    expanded_batch = expanded_batch[triplet_order]
    eq_class_emb_ids = eq_class_emb_ids[triplet_order]
    pred_ids = pred_ids[triplet_order]
    repeated_attn_mask = repeated_attn_mask[triplet_order]
    selected_vocab = selected_vocab[triplet_order]
    independent_entity_separators = [
        independent_entity_separators[i] for i in new_order
    ]
    dependent_entity_separators = [dependent_entity_separators[i] for i in new_order]

    assert (
        expanded_batch.shape[0]
        == eq_class_emb_ids.shape[0]
        == pred_ids.shape[0]
        == repeated_attn_mask.shape[0]
    ), (
        f"Expanded batch shape: {expanded_batch.shape}, eq_class_emb_ids shape: {eq_class_emb_ids.shape}, pred_ids shape: {pred_ids.shape}, repeated_attn_mask shape: {repeated_attn_mask.shape}"
    )

    if insert_negative_samples:
        negatives = construct_negative_samples(
            expanded_batch,
            eq_class_emb_ids,
            vocab_size=output_only_model.model.config.vocab_size,
            fraction=0.5,
        )
        # Append the negatives and duplicate every per-row companion so that
        # each negative row keeps the same metadata as the positive it mirrors.
        expanded_batch = torch.cat([expanded_batch, negatives], dim=0)
        eq_class_emb_ids = eq_class_emb_ids.repeat(2, 1)
        pred_ids = pred_ids.repeat(2, 1)
        repeated_attn_mask = repeated_attn_mask.repeat(2, 1)
        selected_vocab = selected_vocab.repeat(2, 1, 1)
        independent_entity_separators = independent_entity_separators * 2
        dependent_entity_separators = dependent_entity_separators * 2
        new_order = new_order * 2

    # Initialize dataset and dataloader
    local_dataset = TensorDataset(
        expanded_batch.cpu(),
        eq_class_emb_ids.cpu(),
        pred_ids.cpu(),
        repeated_attn_mask.cpu(),
        selected_vocab.cpu(),
    )
    dataloader = DataLoader(
        local_dataset,
        batch_size=minibatch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    final_outputs = defaultdict(list)
    final_outputs["independent_entity_separators"].extend(independent_entity_separators)
    final_outputs["dependent_entity_separators"].extend(dependent_entity_separators)
    final_outputs["new_order"].extend(new_order)

    del selected_inputs, predictions, raw_predictions, token_predictions
    torch.cuda.empty_cache()
    gc.collect()

    for minibatch in tqdm(dataloader, desc=f"Batch on device {local_device}"):
        (
            minibatch_inputs,
            minibatch_eq_class_emb_ids,
            minibatch_pred_ids,
            minibatch_attn_mask,
            minibatch_selected_vocab,
        ) = minibatch
        minibatch_inputs = minibatch_inputs.to(local_device, dtype=torch.int32)
        minibatch_eq_class_emb_ids = [
            row[torch.logical_and(row >= 0, row < minibatch_attn_mask[i].sum())]
            .unique()
            .tolist()
            for i, row in enumerate(minibatch_eq_class_emb_ids)
        ]
        minibatch_pred_ids = [
            row[torch.logical_and(row >= 0, row < minibatch_attn_mask[i].sum())]
            .unique()
            .tolist()
            for i, row in enumerate(minibatch_pred_ids)
        ]
        minibatch_attn_mask = minibatch_attn_mask.to(local_device, dtype=torch.bool)
        minibatch_selected_vocab = minibatch_selected_vocab.to(
            local_device, dtype=torch.int32
        )

        if hasattr(output_only_model.model.base_model, "embeddings"):
            input_embeddings = output_only_model.model.base_model.embeddings(
                minibatch_inputs
            )
        elif hasattr(output_only_model.model.base_model, "embed_tokens"):
            input_embeddings = output_only_model.model.base_model.embed_tokens(
                minibatch_inputs
            )
        else:
            raise AttributeError("Model has neither embeddings nor embed_tokens")
        # input_embeddings.shape = (batch_size*3*N_combinations(len(spans), 2), max_len, embedding_size)

        g = torch.eye(
            selected_vocab.shape[-1],
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )
        g = (
            g.unsqueeze(0)
            .unsqueeze(0)
            .expand(
                input_embeddings.shape[0],
                max(map(len, minibatch_pred_ids))
                * max(map(len, minibatch_eq_class_emb_ids)),
                -1,
                -1,
            )
        )

        (
            pullback_pdets,
            pullback_traces,
            pullback_radii,
            diverging_eigenvectors,
            predictions,
        ) = pullback(
            input_embeddings,
            g=g,
            model=output_only_model,
            eq_class_emb_ids=minibatch_eq_class_emb_ids,
            pred_id=minibatch_pred_ids,
            select=minibatch_selected_vocab,
            attention_mask=minibatch_attn_mask,
            approximated_eigendecomposition=False,
            return_trace=True,
            return_predictions=True,
            min_num_eigenvectors=min_num_eigenvectors,
            max_num_eigenvectors=max_num_eigenvectors,
        )
        tqdm.write(f"Pullback pdets shape: {pullback_pdets.shape}")
        tqdm.write(f"Pullback traces shape: {pullback_traces.shape}")
        tqdm.write(f"Pullback radii shape: {pullback_radii.shape}")
        tqdm.write(f"Diverging eigenvectors shape: {diverging_eigenvectors.shape}")
        # For eigenvectors, we expect dimensions 1 and 3 to change at each minibatch, so we set a minimum number of eigenvectors to take
        if diverging_eigenvectors.shape[-2] >= min_num_eigenvectors:
            diverging_eigenvectors = diverging_eigenvectors[
                ..., :min_num_eigenvectors, :
            ]
        else:
            diverging_eigenvectors = torch.cat(
                [
                    diverging_eigenvectors,
                    torch.zeros(
                        diverging_eigenvectors.shape[0],
                        diverging_eigenvectors.shape[1],
                        min_num_eigenvectors - diverging_eigenvectors.shape[-2],
                        diverging_eigenvectors.shape[-1],
                        device=diverging_eigenvectors.device,
                        dtype=diverging_eigenvectors.dtype,
                    ),
                ],
                dim=-2,
            )

        final_outputs["pdets"].extend([T for T in pullback_pdets.cpu()])
        final_outputs["traces"].extend([T for T in pullback_traces.cpu()])
        final_outputs["radii"].extend([T for T in pullback_radii.cpu()])
        final_outputs["eigenvectors"].extend([T for T in diverging_eigenvectors.cpu()])
        final_outputs["pred_probas"].extend([T for T in predictions.cpu()])

        del (
            minibatch_inputs,
            minibatch_eq_class_emb_ids,
            minibatch_pred_ids,
            minibatch_attn_mask,
            minibatch_selected_vocab,
            input_embeddings,
            g,
        )
        torch.cuda.empty_cache()
        gc.collect()

    pred_probas = final_outputs.pop("pred_probas")
    final_outputs = {
        k: pad_traces_and_eigenvectors(v)
        if isinstance(v[0], torch.Tensor)
        else torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(tup).reshape(-1, 1) for tup in v], batch_first=True
        )
        for k, v in final_outputs.items()
    }

    # Data is already in triplet order: entity_s at [::3], entity_o at [1::3], pair at [2::3]
    final_outputs["independent_entity_separators"] = independent_entity_separators
    final_outputs["dependent_entity_separators"] = dependent_entity_separators
    final_outputs["cond_probas_i"] = pred_probas[::3]
    final_outputs["cond_probas_j"] = pred_probas[1::3]
    final_outputs["mask_probas"] = pred_probas[2::3]
    final_outputs["span_combinations"] = span_combinations
    final_outputs["ent_spans"] = ent_spans
    final_outputs["valid_documents"] = valid_documents
    final_outputs["filtered_batch"] = batch

    return final_outputs


def arnold_gokhale_denominator(
    P_i_cond_j: torch.Tensor,
    P_j_cond_i: torch.Tensor,
    vocab_size: int = None,
    tol: float = 1e-2,
    max_iter: int = 100,
):
    """From 'Distributions most nearly compatible with given families of conditional distributions' by Arnold and Gokhale (1998)"""

    if vocab_size is None:
        vocab_size = P_i_cond_j.shape[-1]

    assert P_i_cond_j.shape[-1] == vocab_size == P_j_cond_i.shape[-1]
    input_shapes = (P_i_cond_j.shape, P_j_cond_i.shape)
    assert input_shapes[0][0] == input_shapes[1][0]
    batch_size = input_shapes[0][0]
    # Input tensor can have multiple batch_shapes, for example if they represent multiple tokens in the same entity, which need to be treated separately and re-aggregated later

    # Reduce precision as this algorithm needs more memory
    algorithm_dtype = torch.float32
    P_i_cond_j = P_i_cond_j.reshape(batch_size, -1, 1, vocab_size, 1).to(
        dtype=algorithm_dtype
    )
    P_j_cond_i = P_j_cond_i.reshape(batch_size, 1, -1, 1, vocab_size).to(
        dtype=algorithm_dtype
    )

    P_i_cond_j = torch.nn.functional.normalize(P_i_cond_j, dim=-2, p=1)
    P_j_cond_i = torch.nn.functional.normalize(P_j_cond_i, dim=-1, p=1)
    idx_i_cond_j = P_i_cond_j.argmax(dim=-2).unsqueeze(-2)
    idx_j_cond_i = P_j_cond_i.argmax(dim=-1).unsqueeze(-1)

    Xi = torch.randn_like(P_i_cond_j).softmax(dim=-2)
    Xj = torch.randn_like(P_j_cond_i).softmax(dim=-1)

    Xij = Xi * Xj
    # Xij.shape = (batch_size, max_len_entities, max_len_entities, selected_vocab_size, selected_vocab_size)
    delta = tol + 1

    for t in range(max_iter):
        inv_sum = torch.reciprocal(
            Xi + torch.finfo(algorithm_dtype).eps
        ) + torch.reciprocal(Xj + torch.finfo(algorithm_dtype).eps)
        num = (P_i_cond_j + P_j_cond_i) / inv_sum
        # num.shape = (batch_size, max_len_entities, max_len_entities, selected_vocab_size, selected_vocab_size)
        den = num.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        # den.shape = (batch_size, max_len_entities, max_len_entities, 1, 1)
        new_Xij = num / (den + EPS)
        delta = torch.norm(new_Xij - Xij, p=1)
        Z = den * inv_sum

        if delta.item() < tol:
            break

        # `new_Xij` is a freshly computed tensor on every iteration, so we can
        # rebind directly instead of deep-copying it.
        Xij = new_Xij
        Xi = Xij.sum(dim=-1, keepdim=True)
        Xj = Xij.sum(dim=-2, keepdim=True)

    if idx_i_cond_j.shape[2] < Z.shape[2]:
        idx_i_cond_j = idx_i_cond_j.repeat_interleave(
            Z.shape[2] // idx_i_cond_j.shape[2], dim=2
        )
    if idx_j_cond_i.shape[1] < Z.shape[1]:
        idx_j_cond_i = idx_j_cond_i.repeat_interleave(
            Z.shape[1] // idx_j_cond_i.shape[1], dim=1
        )

    Z = torch.gather(Z, dim=-2, index=idx_i_cond_j.expand(-1, -1, -1, -1, vocab_size))
    Z = torch.gather(Z, dim=-1, index=idx_j_cond_i)

    back_to_input_shape = (input_shapes[0][0], input_shapes[0][1], input_shapes[1][1])
    Z = Z.reshape(*back_to_input_shape)

    if torch.isnan(Z).any():
        warnings.warn("NaN values in Z")

    P_joints = torch.reshape(
        P_i_cond_j.max(dim=-2).values + P_j_cond_i.max(dim=-1).values,
        back_to_input_shape,
    ) / (Z + EPS)

    return Z, P_joints


def and_operation(x: torch.Tensor, dim: int = -1, eps_correction: float = 1e-6):
    """Exp-sum-log operation to avoid that too small values become 0 in an approximation of minimum or product."""

    max_value = x.max().item()
    mask = torch.logical_and(torch.isfinite(x), x > EPS)
    x = torch.where(mask, x, torch.ones_like(x) - EPS)  # Avoid 0 values
    output = torch.log(x + eps_correction).sum(dim=dim) / (
        torch.sum(mask, dim=dim) + EPS
    )
    output = output.exp()
    output = torch.clamp(output, min=0, max=max_value)
    output = torch.where(mask.sum(dim=dim) > 0, output, torch.zeros_like(output))

    nonfinite_values = (
        torch.logical_or(torch.isnan(output), torch.isinf(output)).sum().item()
    )
    if nonfinite_values > 0:
        print(
            f"Warning: {nonfinite_values} non-finite values in and_operation. Turning them into 0."
        )
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

    return output


@torch.no_grad()
@timing_decorator
def compute_mutual_information(
    traces: torch.Tensor,
    pdets: torch.Tensor,
    radii: torch.Tensor,
    cond_probas_i: torch.Tensor,
    cond_probas_j: torch.Tensor,
    mask_probas: torch.Tensor,
    dependent_entity_separators: torch.Tensor,
    p_minkowski: float = 2,
):

    embedding_size = traces.shape[-1]
    if not isinstance(cond_probas_i, torch.Tensor):
        cond_probas_i = torch.nn.utils.rnn.pad_sequence(cond_probas_i, batch_first=True)
    if not isinstance(cond_probas_j, torch.Tensor):
        cond_probas_j = torch.nn.utils.rnn.pad_sequence(cond_probas_j, batch_first=True)
    if not isinstance(mask_probas, torch.Tensor):
        mask_probas = torch.nn.utils.rnn.pad_sequence(mask_probas, batch_first=True)
    if not isinstance(dependent_entity_separators, torch.Tensor):
        dependent_entity_separators = torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor(tup).reshape(-1, 1)
                for tup in dependent_entity_separators
            ],
            batch_first=True,
        ).to(dtype=torch.int32)  # shape = (batch_size, max_len)

    dependent_entity_separators = dependent_entity_separators.cumsum(dim=1)

    cond_probas_i = torch.nn.functional.normalize(cond_probas_i, dim=-1, p=1)
    cond_probas_j = torch.nn.functional.normalize(cond_probas_j, dim=-1, p=1)
    mask_probas = torch.nn.functional.normalize(mask_probas, dim=-1, p=1)

    cond_predictions_i, cond_pred_indices_i = cond_probas_i.max(dim=-1)
    cond_predictions_j, cond_pred_indices_j = cond_probas_j.max(dim=-1)
    mask_predictions, mask_pred_indices = mask_probas.max(dim=-1)
    batch_size, max_len_entities, vocab_size = cond_probas_i.shape
    # predictions.shape = (N_combinations, max_len_entities)

    cond_predictions_i = cond_predictions_i.squeeze(-1)
    cond_predictions_j = cond_predictions_j.squeeze(-1)
    _dev = mask_probas.device
    positions_of_predictions = (
        torch.arange(mask_predictions.shape[-1], device=_dev)
        .reshape(1, -1)
        .expand(mask_predictions.shape[0], -1)
    )
    mask_Pi = torch.where(
        positions_of_predictions < dependent_entity_separators[::3, 0], 1, 0
    ).unsqueeze(-1)
    mask_Pj_num = torch.where(
        positions_of_predictions < dependent_entity_separators[1::3, 1], 1, 0
    ).unsqueeze(-1)
    mask_Pj_den = torch.where(
        torch.logical_and(
            positions_of_predictions >= dependent_entity_separators[2::3, 0],
            positions_of_predictions < dependent_entity_separators[2::3, 1],
        ),
        1,
        0,
    ).unsqueeze(-1)

    # cond_probas.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities, selected_vocab_size)
    vocab_Pi = cond_probas_i * mask_Pi[:, : cond_probas_i.shape[1], :]
    vocab_Pj = cond_probas_j * mask_Pj_num[:, : cond_probas_j.shape[1], :]
    num_Pi = (
        cond_predictions_i.unsqueeze(-1) * mask_Pi[:, : cond_predictions_i.shape[1], :]
    )
    num_Pj = (
        cond_predictions_j.unsqueeze(-1)
        * mask_Pj_num[:, : cond_predictions_j.shape[1], :]
    )
    num_Pj = num_Pj.transpose(-2, -1)
    den_Pi = mask_predictions.unsqueeze(-1) * mask_Pi[:, : mask_predictions.shape[1], :]
    den_Pj = (
        mask_predictions.unsqueeze(-1) * mask_Pj_den[:, : mask_predictions.shape[1], :]
    )
    den_Pj = den_Pj.transpose(-2, -1)
    max_P_shape = max(
        den_Pj.shape[2], den_Pi.shape[1], num_Pi.shape[1], num_Pj.shape[2]
    )
    den_Pj = torch.nn.functional.pad(den_Pj, (0, max_P_shape - den_Pj.shape[2]))
    den_Pi = torch.nn.functional.pad(den_Pi, (0, 0, 0, max_P_shape - den_Pi.shape[1]))
    num_Pi = torch.nn.functional.pad(num_Pi, (0, 0, 0, max_P_shape - num_Pi.shape[1]))
    num_Pj = torch.nn.functional.pad(num_Pj, (0, max_P_shape - num_Pj.shape[2]))
    # Pi.shape = (N_combinations, max_len_entities, 1)
    # Pj.shape = (N_combinations, 1, max_len_entities)

    Zs, P_joints = arnold_gokhale_denominator(vocab_Pi, vocab_Pj, vocab_size=vocab_size)
    # Zs.shape = (N_combinations, max_len_entities, max_len_entities)

    Zs = torch.nn.functional.pad(
        Zs, (0, max_P_shape - Zs.shape[-1], 0, max_P_shape - Zs.shape[-2]), value=1
    )
    P_joints = torch.nn.functional.pad(
        P_joints,
        (0, max_P_shape - P_joints.shape[-1], 0, max_P_shape - P_joints.shape[-2]),
        value=0,
    )
    alpha_i = (1 + num_Pi / (num_Pj + EPS)) / (Zs + EPS)
    alpha_j = (1 + num_Pj / (num_Pi + EPS)) / (Zs + EPS)
    # alpha.shape = (N_combinations, max_len_entities, max_len_entities)

    internal_log = torch.log(
        (alpha_i * den_Pi + alpha_j * den_Pj + EPS) / (den_Pi**2 + den_Pj**2 + EPS)
    )
    internal_term_j = num_Pi - den_Pi
    internal_term_i = num_Pj - den_Pj
    Ci = alpha_i**2 * ((internal_log + internal_term_i) ** 2 - 1) / (Zs**2 + EPS)
    Cj = alpha_j**2 * ((internal_log + internal_term_j) ** 2 - 1) / (Zs**2 + EPS)

    # C*.shape = (N_combinations, max_len_entities, max_len_entities)

    I_stretch = radii[::2] / traces[::2]
    J_stretch = radii[1::2] / traces[1::2]
    I_stretch = torch.where(
        torch.isfinite(I_stretch), I_stretch, torch.ones_like(I_stretch)
    )
    J_stretch = torch.where(
        torch.isfinite(J_stretch), J_stretch, torch.ones_like(J_stretch)
    )

    pdets_i = pdets[::2]
    pdets_j = pdets[1::2]

    for i, (s1, s2) in enumerate(zip(Ci.shape, I_stretch.shape)):
        if s1 < s2:
            I_stretch = torch.take_along_dim(
                I_stretch,
                indices=torch.arange(s1, device=I_stretch.device).reshape(
                    *[1 if j != i else -1 for j in range(I_stretch.dim())]
                ),
                dim=i,
            )
            pdets_i = torch.take_along_dim(
                pdets_i,
                indices=torch.arange(s1, device=pdets_i.device).reshape(
                    *[1 if j != i else -1 for j in range(pdets_i.dim())]
                ),
                dim=i,
            )
        elif s1 > s2:
            Ci = torch.take_along_dim(
                Ci,
                indices=torch.arange(s2, device=Ci.device).reshape(
                    *[1 if j != i else -1 for j in range(Ci.dim())]
                ),
                dim=i,
            )

    for i, (s1, s2) in enumerate(zip(Cj.shape, J_stretch.shape)):
        if s1 < s2:
            J_stretch = torch.take_along_dim(
                J_stretch,
                indices=torch.arange(s1, device=J_stretch.device).reshape(
                    *[1 if j != i else -1 for j in range(J_stretch.dim())]
                ),
                dim=i,
            )
            pdets_j = torch.take_along_dim(
                pdets_j,
                indices=torch.arange(s1, device=pdets_j.device).reshape(
                    *[1 if j != i else -1 for j in range(pdets_j.dim())]
                ),
                dim=i,
            )
        elif s1 > s2:
            Cj = torch.take_along_dim(
                Cj,
                indices=torch.arange(s2, device=Cj.device).reshape(
                    *[1 if j != i else -1 for j in range(Cj.dim())]
                ),
                dim=i,
            )

    pad_mask = torch.logical_and(
        num_Pi.expand(-1, -1, max_P_shape) > 0,
        num_Pj.expand(-1, max_P_shape, -1) > 0,
    )

    I_term = Ci * torch.sqrt(pdets_i * I_stretch)
    J_term = Cj * torch.sqrt(pdets_j * J_stretch)
    I_term = torch.where(pad_mask, I_term, torch.zeros_like(I_term))
    J_term = torch.where(pad_mask, J_term, torch.zeros_like(J_term))

    I_term_flat = torch.flatten(I_term, start_dim=1)
    J_term_flat = torch.flatten(J_term, start_dim=1)
    I_term_mean = I_term_flat.nansum(dim=-1) / (
        torch.count_nonzero(I_term_flat, dim=-1) + EPS
    )
    J_term_mean = J_term_flat.nansum(dim=-1) / (
        torch.count_nonzero(J_term_flat, dim=-1) + EPS
    )
    I_term_frob = (
        (I_term_flat / (torch.count_nonzero(I_term_flat, dim=-1) + EPS).unsqueeze(-1))
        .pow(2)
        .nansum(dim=-1)
        .sqrt()
    )
    J_term_frob = (
        (J_term_flat / (torch.count_nonzero(J_term_flat, dim=-1) + EPS).unsqueeze(-1))
        .pow(2)
        .nansum(dim=-1)
        .sqrt()
    )
    I_term_andin_orout = and_operation(
        I_term.pow(p_minkowski).nanmean(dim=-2).pow(1 / p_minkowski), dim=-1
    )
    J_term_andin_orout = and_operation(
        J_term.pow(p_minkowski).nanmean(dim=-1).pow(1 / p_minkowski), dim=-1
    )
    I_term_andout_orin = and_operation(
        I_term.pow(p_minkowski).nanmean(dim=-1).pow(1 / p_minkowski), dim=-1
    )
    J_term_andout_orin = and_operation(
        J_term.pow(p_minkowski).nanmean(dim=-2).pow(1 / p_minkowski), dim=-1
    )

    smits_mean = 2 * math.sqrt(embedding_size * 2) * (I_term_mean + J_term_mean)
    smits_frob = 2 * math.sqrt(embedding_size * 2) * (I_term_frob + J_term_frob)
    smits_andin_orout = (
        2 * math.sqrt(embedding_size * 2) * (I_term_andin_orout + J_term_andin_orout)
    )
    smits_andout_orin = (
        2 * math.sqrt(embedding_size * 2) * (I_term_andout_orin + J_term_andout_orin)
    )
    pointwise_mi = (
        torch.log((P_joints + EPS) / (den_Pi * den_Pj.transpose(-2, -1) + EPS))
        .flatten(start_dim=1)
        .mean(dim=-1)
    )

    for n, T in enumerate(
        [smits_mean, smits_frob, smits_andin_orout, smits_andout_orin, pointwise_mi]
    ):
        if torch.isnan(T).any():
            warnings.warn(f"NaN values in scoring metric number {n}")
            T = T.nan_to_num(0)

    return {
        "smits_mean": smits_mean,
        "smits_frob": smits_frob,
        "smits_andin_orout": smits_andin_orout,
        "smits_andout_orin": smits_andout_orin,
        "pointwise_mi": pointwise_mi,
    }


def get_combination_indices_with_repeats(lists, num_indices_per_doc=None):
    """
    Returns the sequence of combination indices for each document in ex.
    After every 2 tuples, inserts the first tuple (tup0) again.

    Args:
        ex: List of documents
        num_indices_per_doc: Optional list of number of indices per document.
                            If None, uses len(doc) for each document if it's a list/str,
                            otherwise uses a default value.

    Returns:
        List of lists of tuples: [[(0,1), (0,2), (0,1), (0,3), (1,2), (0,1), ...], ...]
        One sequence per document.
    """
    if num_indices_per_doc is None:
        num_indices_per_doc = [len(doc) for doc in lists]

    result = []
    for num_indices in num_indices_per_doc:
        # Generate all combinations of 2 indices.
        combos = combinations(range(num_indices), 2)

        # Each combo is repeated 3 times (entity_s, entity_o, pair). Use
        # `chain.from_iterable` to flatten in linear time; `sum([], [])` was
        # quadratic in the number of pairs.
        doc_result = [
            list(combo) for combo in chain.from_iterable((c, c, c) for c in combos)
        ]

        result.extend(doc_result)

    return result


@timing_decorator
def main(
    test: bool = False,
    model_name: str = None,
    with_mbrd: bool = False,
    with_baselines: bool = False,
    devices: List[str] = None,
):

    print(f"Scoring model: {model_name}")
    if devices is not None:
        train_device = torch.device(devices[0])
    else:
        train_device = _resolve_torch_device(None)

    training_data = load_dataset("xiaobendanyn/tacred", split="train")
    data = load_dataset("xiaobendanyn/tacred", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "distilbert" in model_name.lower():
        encoder_name = "transformer"
    elif "electra" in model_name.lower():
        encoder_name = ["embeddings_project", "encoder"]
    elif "llama" in model_name.lower():
        encoder_name = "layers"
    elif "phi" in model_name.lower():
        encoder_name = "layers"
    else:
        encoder_name = "encoder"

    train_df = pd.DataFrame([json.loads(d["text"]) for d in training_data]).map(
        lambda x: (
            x["name"]
            if isinstance(x, dict)
            else " ".join(x)
            if isinstance(x, list)
            else x
        )
    )
    df = pd.DataFrame([json.loads(d["text"]) for d in data]).map(
        lambda x: (
            x["name"]
            if isinstance(x, dict)
            else " ".join(x)
            if isinstance(x, list)
            else x
        )
    )
    df.rename(columns={"token": "input", "relation": "label"}, inplace=True)

    # Tokenize every sentence once and reuse the BatchEncoding's char_to_token
    # mapping, instead of re-tokenizing per row (and per query).
    df["batch_encodings"] = tokenizer(
        df["input"].tolist(), add_special_tokens=False, padding=False
    ).encodings
    df["h_token_spans"] = df.apply(
        lambda row: _find_tokens(row["input"], row["h"], row["batch_encodings"]), axis=1
    )
    df["t_token_spans"] = df.apply(
        lambda row: _find_tokens(row["input"], row["t"], row["batch_encodings"]), axis=1
    )
    df["gold_spans"] = df[["h_token_spans", "t_token_spans"]].apply(
        lambda x: x["h_token_spans"] + x["t_token_spans"], axis=1
    )
    train_df.rename(columns={"token": "input", "relation": "label"}, inplace=True)

    model_config = hf_api.model_info(model_name).config
    hidden_size = getattr(model_config, "hidden_size", 0)
    if hidden_size > 1024:
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    with open("hftoken", "r") as f:
        hf_token = f.read().strip()

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    ).to(train_device, dtype=torch.get_default_dtype())
    print("Model loaded", flush=True)
    batch_size = 12
    max_num_eigenvectors = (
        model.config.embedding_size // 2
        if hasattr(model.config, "embedding_size")
        else model.config.hidden_size // 2
    )
    max_retained_vocab = 50
    os.makedirs("../results", exist_ok=True)

    if test:
        df = df.sample(16)

    df.reset_index(inplace=True)
    ex = [row for _, row in df.iterrows()]
    random.shuffle(ex)
    # train_ex = train_df["input"].tolist()
    full_results = []
    # full_inputs = tokenizer(train_ex, padding = "longest", add_special_tokens = False, truncation=True, return_tensors = "pt")

    # mlm_model = train_lm_head(model_name, train_device, mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.pad_token_id, n_epochs = 50, inputs = full_inputs.input_ids, attn_mask = full_inputs.attention_mask, encoder_name = encoder_name, hf_token = hf_token)
    mlm_model = model
    max_retained_vocab = min(max_retained_vocab, len(model.config.id2label))

    if devices is not None and len(devices) > 1:
        mlm_by_device = {}
        ner_by_device = {}
        for i, device in enumerate(devices):
            d = torch.device("cuda", i)
            if d == train_device:
                mlm_by_device[d] = mlm_model
                ner_by_device[d] = model
            else:
                mlm_by_device[d] = deepcopy(mlm_model).to(d)
                ner_by_device[d] = deepcopy(model).to(d)
    else:
        mlm_by_device = {train_device: mlm_model}
        ner_by_device = {train_device: model}

    def score_batch(
        batch: List[str],
        work_device: torch.device = None,
        mlm_for_pass=None,
        ner_for_pass=None,
        gold_spans: List[List[Tuple[int, int]]] = None,
        doc_indices: List[int] = None,
    ):
        torch.cuda.empty_cache()
        gc.collect()

        work_device = (
            train_device if work_device is None else _resolve_torch_device(work_device)
        )
        mlm_use = mlm_by_device[work_device] if mlm_for_pass is None else mlm_for_pass
        ner_use = ner_by_device[work_device] if ner_for_pass is None else ner_for_pass
        if getattr(work_device, "type", None) == "cuda":
            torch.cuda.set_device(work_device)

        pullback_dict = forward_backward_pass(
            batch,
            ner_use,
            tokenizer,
            mlm_use,
            gold_spans=gold_spans,
            max_num_eigenvectors=max_num_eigenvectors,
            max_retained_vocab=max_retained_vocab,
            encoder_name=encoder_name,
        )
        # In the dictionary, pred_probas and mask_probas have 1/3 of the size of the other tensors

        if len(pullback_dict) == 0:
            print("No valid results found for batch", flush=True)
            return [{}] * sum(map(len, batch))

        entity_span_combinations = pullback_dict.pop("span_combinations")
        entity_spans = pullback_dict.pop("ent_spans")
        valid_document_numbers = pullback_dict.pop("valid_documents")
        if len(valid_document_numbers) < len(entity_span_combinations):
            entity_span_combinations = [
                entity_span_combinations[i] for i in valid_document_numbers
            ]
        valid_document_numbers = [
            doc_indices[i] if doc_indices is not None else i
            for i in valid_document_numbers
        ]
        filtered_batch = pullback_dict.pop("filtered_batch")
        dependent_entity_separators = pullback_dict.pop("dependent_entity_separators")
        total_combinations = sum(len(combos) for combos in entity_span_combinations)

        if any(
            res is None
            for res in [
                pullback_dict,
                entity_span_combinations,
                entity_spans,
                valid_document_numbers,
            ]
        ):
            print("No valid results found for batch", flush=True)
            return [{}] * sum(map(len, batch))

        # Convert separators to a tensor of shape (batch_size, max_len, 1)
        dependent_entity_separators = torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor(tup).reshape(-1, 1)
                for tup in dependent_entity_separators
            ],
            batch_first=True,
        ).to(dtype=torch.int32)  # shape = (batch_size, max_len)

        mi_dict = compute_mutual_information(
            pullback_dict["traces"],
            pullback_dict["pdets"],
            pullback_dict["radii"],
            pullback_dict["cond_probas_i"],
            pullback_dict["cond_probas_j"],
            pullback_dict["mask_probas"],
            dependent_entity_separators,
        )

        mi = mi_dict["smits_mean"].detach().cpu()  # shape = (total_combinations,)
        mi_frob = mi_dict["smits_frob"].detach().cpu()  # shape = (total_combinations,)
        pointwise_mi = (
            mi_dict["pointwise_mi"].detach().cpu()
        )  # shape = (total_combinations,)
        smits_andin_orout = (
            mi_dict["smits_andin_orout"].detach().cpu()
        )  # shape = (total_combinations,)
        smits_andout_orin = (
            mi_dict["smits_andout_orin"].detach().cpu()
        )  # shape = (total_combinations,)

        entity_eigenvectors = pullback_dict.pop("eigenvectors").detach().cpu()
        entity_eigenvectors = entity_eigenvectors.reshape(
            total_combinations, -1, *entity_eigenvectors.shape[-3:]
        )  # shape = (total_combinations, 2*num_tokens_in_i, 2*num_tokens_in_j, num_eigenvectors, embedding_size)
        radii = (
            pullback_dict.pop("radii")
            .detach()
            .cpu()
            .nan_to_num(nan=0, posinf=1, neginf=-1)
        )  # shape = (2*total_combinations, num_tokens_in_i, num_tokens_in_j)
        traces = (
            pullback_dict.pop("traces")
            .detach()
            .cpu()
            .nan_to_num(nan=0, posinf=1, neginf=-1)
        )  # shape = (2*total_combinations, num_tokens_in_i, num_tokens_in_j)
        pdets = (
            pullback_dict.pop("pdets")
            .detach()
            .cpu()
            .nan_to_num(nan=0, posinf=1, neginf=-1)
        )  # shape = (2*total_combinations, num_tokens_in_i, num_tokens_in_j)
        cond_probas_i = pullback_dict.pop(
            "cond_probas_i"
        )  # shape = (total_combinations, num_tokens_in_i, selected_vocab_size)
        cond_probas_i = torch.nn.utils.rnn.pad_sequence(
            [T.detach().cpu() for T in cond_probas_i], batch_first=True
        ).nan_to_num(
            nan=0, posinf=1, neginf=0
        )  # shape = (total_combinations, num_tokens_in_i, selected_vocab_size)
        cond_probas_j = pullback_dict.pop(
            "cond_probas_j"
        )  # shape = (total_combinations, num_tokens_in_j, selected_vocab_size)
        cond_probas_j = torch.nn.utils.rnn.pad_sequence(
            [T.detach().cpu() for T in cond_probas_j], batch_first=True
        ).nan_to_num(
            nan=0, posinf=1, neginf=0
        )  # shape = (total_combinations, num_tokens_in_j, selected_vocab_size)
        mask_probas = pullback_dict.pop(
            "mask_probas"
        )  # shape = (total_combinations, num_tokens_in_i, num_tokens_in_j, selected_vocab_size)
        mask_probas = torch.nn.utils.rnn.pad_sequence(
            [T.detach().cpu() for T in mask_probas], batch_first=True
        ).nan_to_num(
            nan=0, posinf=1, neginf=0
        )  # shape = (total_combinations, num_tokens_in_i, num_tokens_in_j, selected_vocab_size)
        del pullback_dict, ner_use
        torch.cuda.empty_cache()
        gc.collect()

        # Mean Integrated Directional Gradients baseline
        if with_baselines:
            mean_idg_scores = mean_integrated_directional_gradients(
                mlm_use,
                filtered_batch,
                tokenizer,
                entity_span_combinations,
                M=9,
                encoder_name=encoder_name,
            )
            shap_iq_scores_single = [
                tup[0]
                for tup in shapley_interaction_quantification(
                    mlm_use,
                    filtered_batch,
                    tokenizer,
                    entity_spans,
                    encoder_name=encoder_name,
                )
            ]
            shap_iq_scores = [
                (single_scores[i] + single_scores[j]) / 2
                for single_scores in shap_iq_scores_single
                for i, j in combinations(range(len(single_scores)), 2)
            ]

            mlm_use = mlm_use.to(work_device)
            mlm_use.requires_grad_(True)

        current_pos = 0
        batch_results = []
        retokenized_batch = tokenizer(
            filtered_batch,
            padding="max_length",
            add_special_tokens=False,
            max_length=model.config.max_position_embeddings - 2
            if hasattr(model.config, "max_position_embeddings")
            else model.config.n_ctx - 2,
            truncation=True,
            return_tensors="pt",
        )
        embedder = (
            mlm_use.base_model.embeddings
            if hasattr(mlm_use.base_model, "embeddings")
            else mlm_use.base_model.embed_tokens
        )
        with torch.no_grad():
            entity_embeddings = embedder(
                retokenized_batch.input_ids.to(work_device)
            ).cpu()

        # Pre-tokenize each sentence once instead of re-tokenizing per (entity, span) pair below.
        batch_token_lists = [
            tokenizer.tokenize(s, add_special_tokens=False) for s in filtered_batch
        ]

        # But eigenvectors, radii and traces are not 1:1 with entity_span_combinations, so we need to re-index them
        for i, sent_span_combinations in enumerate(entity_span_combinations):
            k = len(sent_span_combinations)

            sent_tokens = batch_token_lists[i]
            entity_pairs_scores = [
                {
                    "e_i_idx": list(span_combo[0]),
                    "e_j_idx": list(span_combo[1]),
                    "e_i_tokens": " ".join(
                        sent_tokens[span_combo[0][0] : span_combo[0][1]]
                    ),
                    "e_j_tokens": " ".join(
                        sent_tokens[span_combo[1][0] : span_combo[1][1]]
                    ),
                    "score_mean": mi[current_pos + j]
                    .to("cpu", dtype=torch.float64)
                    .item(),
                    "score_frob": mi_frob[current_pos + j]
                    .to("cpu", dtype=torch.float64)
                    .item(),
                    "score_andin_orout": smits_andin_orout[current_pos + j]
                    .to("cpu", dtype=torch.float64)
                    .item(),
                    "score_andout_orin": smits_andout_orin[current_pos + j]
                    .to("cpu", dtype=torch.float64)
                    .item(),
                    "pointwise_baseline": pointwise_mi[current_pos + j]
                    .to("cpu", dtype=torch.float64)
                    .item(),
                    "mean_idg_baseline": mean_idg_scores[current_pos + j]
                    .to("cpu", dtype=torch.float64)
                    .item()
                    if with_baselines
                    else -1.0,
                    "shap_iq_baseline": shap_iq_scores[current_pos + j]
                    if with_baselines
                    else -1.0,
                    "original_text": filtered_batch[i],
                    "doc_number": valid_document_numbers[i],
                    "e_i_embeddings": entity_embeddings[i][
                        span_combo[0][0] : span_combo[0][1]
                    ].to("cpu", dtype=torch.float64),
                    "e_i_eigenvectors": entity_eigenvectors[i][
                        : dependent_entity_separators[current_pos + j][0]
                    ].to("cpu", dtype=torch.float64),
                    "e_i_probas_joint": cond_probas_i[current_pos + j][
                        : dependent_entity_separators[current_pos + j][0]
                    ].to("cpu", dtype=torch.float64),
                    "e_j_probas_joint": cond_probas_j[current_pos + j][
                        dependent_entity_separators[current_pos + j][0] :
                    ].to("cpu", dtype=torch.float64),
                    "e_i_probas_prod": mask_probas[current_pos + j][
                        : dependent_entity_separators[current_pos + j][0]
                    ].to("cpu", dtype=torch.float64),
                    "e_j_probas_prod": mask_probas[current_pos + j][
                        dependent_entity_separators[current_pos + j][0] :
                    ].to("cpu", dtype=torch.float64),
                    "e_i_pdets": pdets[current_pos + j][
                        : dependent_entity_separators[current_pos + j][0]
                    ].to("cpu", dtype=torch.float64),
                    "e_j_pdets": pdets[current_pos + j][
                        dependent_entity_separators[current_pos + j][0] :
                    ].to("cpu", dtype=torch.float64),
                    "e_i_traces": traces[current_pos + j][
                        : dependent_entity_separators[current_pos + j][0]
                    ].to("cpu", dtype=torch.float64),
                    "e_j_traces": traces[current_pos + j][
                        dependent_entity_separators[current_pos + j][0] :
                    ].to("cpu", dtype=torch.float64),
                    "e_i_radii": radii[current_pos + j][
                        : dependent_entity_separators[current_pos + j][0]
                    ].to("cpu", dtype=torch.float64),
                    "e_j_radii": radii[current_pos + j][
                        dependent_entity_separators[current_pos + j][0] :
                    ].to("cpu", dtype=torch.float64),
                    "e_j_embeddings": entity_embeddings[i][
                        span_combo[1][0] : span_combo[1][1]
                    ].to("cpu", dtype=torch.float64),
                    "e_j_eigenvectors": entity_eigenvectors[i][
                        dependent_entity_separators[current_pos + j][0] :
                    ].to("cpu", dtype=torch.float64),
                }
                for j, span_combo in enumerate(sent_span_combinations)
            ]

            batch_results.extend(entity_pairs_scores)
            current_pos += k

        return batch_results

    def mbr_batch(
        batch: List[str],
        relation_disambiguation_model: RelationDisambiguation,
        entity_embeddings: List[List[torch.Tensor]],
        entity_eigenvectors: List[List[torch.Tensor]],
        step_sizes: List[List[float | torch.Tensor]],
        num_pseudorefs: int = 10,
        work_device: torch.device = None,
    ):
        work_device = (
            train_device if work_device is None else _resolve_torch_device(work_device)
        )
        if getattr(work_device, "type", None) == "cuda":
            torch.cuda.set_device(work_device)

        # Each entity embedding is a tensor of size (len_entity, embedding_size)
        # Each entity eigenvector is a tensor of size (len_entity, embedding_size, num_eigenvectors)
        # Each entity eigenvalue is a tensor of size (len_entity, num_eigenvectors)

        entity_eigenvectors = [
            [
                T[..., -num_pseudorefs:]
                if num_pseudorefs >= T.shape[-1]
                else torch.nn.functional.pad(T, (0, num_pseudorefs - T.shape[-1]))
                for T in L
            ]
            for L in entity_eigenvectors
        ]

        entity_embeddings = [
            [
                emb.unsqueeze(-1).expand(-1, -1, vec.shape[-1]) + vec * step_size
                for emb, vec, step_size in zip(L1, L2, L3)
            ]
            for L1, L2, L3 in zip(entity_embeddings, entity_eigenvectors, step_sizes)
        ]

        # HYpotheses are already in the relation disambiguation model
        relation_scores = relation_disambiguation_model(entity_embeddings, batch)

        return [relation_scores[i].as_dict() for i in range(len(relation_scores))]

    def score_batch_list(
        batch_list: List[List[str]],
        work_device: torch.device = None,
        mlm_for_pass=None,
        ner_for_pass=None,
        gold_spans: List[List[Tuple[int, int]]] = None,
        doc_indices: List[List[int]] = None,
    ):

        return list(
            chain(
                *[
                    score_batch(
                        batch,
                        work_device=work_device,
                        mlm_for_pass=mlm_for_pass,
                        ner_for_pass=ner_for_pass,
                        gold_spans=gold_spans,
                        doc_indices=doc_indices[i] if doc_indices is not None else None,
                    )
                    for i, batch in enumerate(batch_list)
                ]
            )
        )

    #########################################################
    if devices is not None and len(devices) > 1:
        devices = [torch.device(device) for device in devices]
        batches = list(batch_generator(ex, batch_size))
        list_size = (len(batches) + len(devices) - 1) // len(devices)
        batch_lists = [
            batches[i : i + list_size] for i in range(0, len(batches), list_size)
        ]

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(
                    score_batch_list,
                    [[row["input"] for row in batch] for batch in batch_lists],
                    i * batch_size * list_size,
                    devices[i % len(devices)],
                    mlm_for_pass=mlm_by_device[devices[i % len(devices)]],
                    ner_for_pass=ner_by_device[devices[i % len(devices)]],
                    gold_spans=[
                        [row["gold_spans"] for row in batch] for batch in batch_lists
                    ],
                    doc_indices=[
                        [row["index"] for row in batch] for batch in batch_lists
                    ],
                )
                for i, batch_list in enumerate(batch_lists)
            ]

            for res in as_completed(futures):
                full_results.extend(res.result())

            print(f"Processed {len(full_results)} examples", flush=True)
            results_to_save = [
                {k: v for k, v in d.items() if not isinstance(v, torch.Tensor)}
                for d in full_results
            ]
            with open(
                f"../results/tacred_scores_{model_name.replace('/', '___')}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(results_to_save, f)
    else:
        for i, batch in enumerate(batch_generator(ex, batch_size)):
            full_results.extend(
                score_batch(
                    [row["input"] for row in batch],
                    work_device=train_device,
                    mlm_for_pass=mlm_by_device[train_device],
                    ner_for_pass=ner_by_device[train_device],
                    gold_spans=[row["gold_spans"] for row in batch],
                    doc_indices=[row["index"] for row in batch],
                )
            )
            print(f"Processed {len(full_results)} examples", flush=True)

            results_to_save = [
                {
                    k: v.numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()
                    if k
                    not in [
                        "e_i_embeddings",
                        "e_j_embeddings",
                        "e_i_eigenvectors",
                        "e_j_eigenvectors",
                    ]
                }
                for d in full_results
            ]

            with open(
                f"../results/tacred_scores_{model_name.replace('/', '___')}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(results_to_save, f)

    # MBR DECODING
    full_results_mbrd = []
    if with_mbrd:
        hypotheses = df["label"].unique().tolist() + ["no relation"]
        relation_disambiguation_model = RelationDisambiguation(
            model_name,
            hypotheses,
            topk=5,
            batch_size=batch_size,
            embedding_module_name="embed_tokens"
            if hasattr(mlm_by_device[train_device].model.base_model, "embed_tokens")
            else "embeddings",
            encoder_module_name=encoder_name,
            classifier_module_name="lm_head"
            if hasattr(mlm_by_device[train_device].model.base_model, "lm_head")
            else "generator_lm_head"
            if hasattr(
                mlm_by_device[train_device].model.base_model, "generator_lm_head"
            )
            else "classifier",
        )
        grouped_results = list(
            itertools.groupby(full_results, key=lambda x: x["doc_number"])
        )
        grouped_results = [list(group) for _, group in grouped_results]

        for i, batch in tqdm(
            enumerate(batch_generator(grouped_results, batch_size)),
            desc="MBR Decoding",
            total=len(grouped_results) + batch_size - 1 // batch_size,
        ):
            texts = [L[0]["original_text"] for L in batch]
            entity_embeddings = [[d["entity_embeddings"] for d in L] for L in batch]
            entity_eigenvectors = [[d["entity_eigenvectors"] for d in L] for L in batch]
            step_sizes = [[d["step_size"] for d in L] for L in batch]
            full_results_mbrd.extend(
                mbr_batch(
                    texts,
                    relation_disambiguation_model,
                    entity_embeddings,
                    entity_eigenvectors,
                    step_sizes,
                    num_pseudorefs=10,
                    work_device=train_device,
                )
            )

            print(f"Processed {len(full_results_mbrd)} examples", flush=True)
            with open(
                f"../results/tacred_scores_{model_name.replace('/', '___')}_mbrd.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(full_results_mbrd, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--model-name", type=str, default="tanaos/tanaos-NER-v1")
    parser.add_argument("--mbrd", action="store_true", default=False)
    parser.add_argument("--baselines", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    devices = args.device.split("-") if args.device is not None else None
    main(
        test=args.test,
        model_name=args.model_name,
        with_mbrd=args.mbrd,
        with_baselines=args.baselines,
        devices=devices,
    )
