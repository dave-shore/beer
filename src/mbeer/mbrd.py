from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from mbeer.utils import batch_generator
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import ot.unbalanced as otu
import numpy as np
from mbrs.metrics import Metric, MetricCacheable
from mbrs.decoders import DecoderProbabilisticMBR, DecoderMBR
from mbrs import timer, functional
from mbrs.modules.als import MatrixFactorizationALS
from itertools import product as cartesian
import math


class QKScore(torch.nn.Module):
    def __init__(self, score_statistic: Callable = lambda x: x.quantile(0.95, dim = -1), tokenizer: AutoTokenizer = None, model: AutoModel = None, device: str = "cpu"):
        super().__init__()
        self.score_statistic = score_statistic
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def forward(self, triples: List[Tuple[str,str,str]]) -> float:

        neg_inv_triples = [(triple[2], "NOT " + triple[1], triple[0]) for triple in triples]
        pseudosentences = [[
            " ".join(triple),
            "implies",
            " ".join(neg_inv_triple),
        ] for triple, neg_inv_triple in zip(triples, neg_inv_triples)]

        tokenized_sentences = self.tokenizer(pseudosentences, add_special_tokens = False, return_tensors = "pt", is_split_into_words = True)
        imply_positions = [encoding.word_to_token[1][0] for encoding in tokenized_sentences.encodings]
        pos_positions = [encoding.word_to_token[0] for encoding in tokenized_sentences.encodings]
        max_len_pos_positions = max(len(pos_positions) for pos_positions in pos_positions)
        pos_positions = [pos_positions + [0]*(max_len_pos_positions - len(pos_positions)) for pos_positions in pos_positions]
        neg_positions = [encoding.word_to_token[2] for encoding in tokenized_sentences.encodings]
        max_len_neg_positions = max(len(neg_positions) for neg_positions in neg_positions)
        neg_positions = [neg_positions + [-1]*(max_len_neg_positions - len(neg_positions)) for neg_positions in neg_positions]

        with torch.inference_mode():
            _, attentions = self.model(**tokenized_sentences.to(self.device), output_attentions = True)
            stacked_attentions = torch.stack([A for tup in attentions for A in tup], dim = 1)
            # stacked_attentions.shape = (batch_size, num_layers*num_heads, seq_len, seq_len)
            qk_pos_scores = stacked_attentions[:, :, imply_positions, pos_positions].flatten(start_dim = 2).mean(dim = 2)
            qk_neg_scores = stacked_attentions[:, :, imply_positions, neg_positions].flatten(start_dim = 2).mean(dim = 2)
            qk_scores = qk_pos_scores - qk_neg_scores
            score = self.score_statistic(qk_scores)
            # qk_scores.shape = (batch_size, num_layers*num_heads)
            # score.shape = (batch_size,)
        
        if score.dim() > 1:
            print(f"Warning: score has {score.dim()} dimensions, expected 1")
            score = score.flatten(start_dim = 1).mean(dim = 1)

        return score


class EmbeddingSearchLayer(torch.nn.Module):
    def __init__(self, vectors: torch.Tensor, p: int = 2):
        super().__init__()
        self.vectors = vectors
        self.p = p

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        return torch.cdist(query, self.vectors, p = self.p).argmin(dim = 1)


class DePositionLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Embedding):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.layer.weight[:x.shape[0]]


class DeTypingLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Embedding):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor, token_type: int) -> torch.Tensor:
        return x - self.layer.weight[:x.shape[0], [token_type]]


class DeNormLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.LayerNorm):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer.elementwise_affine:
            y = (x - self.layer.bias) / (self.layer.weight + 1e-6)
        else:
            y = x

        if getattr(self.layer, "running_mean", None) is not None and getattr(self.layer, "running_var", None) is not None:
            y = y * torch.sqrt(self.layer.running_var + 1e-6) + self.layer.running_mean

        return y


class UnEmbedder(torch.nn.Module):
    def __init__(self, embedding_layer: torch.nn.Module, vocab_size: int, context_size: int, token_types: int = 1, p: int = 2):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.p = p
        self.unembedders = []
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.token_types = token_types

        for module in self.embedding_layer.modules():
            if isinstance(module, torch.nn.Linear):
                new_layers = [
                    torch.nn.Linear(module.out_features, module.in_features, bias = False),
                    torch.nn.Linear(module.out_features, module.out_features, bias = True),
                ]
                new_layers[0].weight = torch.linalg.pinv(module.weight)
                new_layers[1].weight = torch.eye(module.out_features)
                new_layers[1].bias = - module.bias
                self.unembedders.extend(new_layers)

            elif isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == self.vocab_size:
                    self.unembedders.append(EmbeddingSearchLayer(module.weight, p = self.p))
                elif module.weight.shape[0] == self.context_size:
                    self.unembedders.append(DePositionLayer(module))
                else:
                    self.unembedders.append(DeTypingLayer(module))

            elif isinstance(module, torch.nn.LayerNorm):
                self.unembedders.append(DeNormLayer(module))

        self.unembedding = torch.nn.Sequential(*self.unembedders[::-1])

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.unembedding(embeddings)


class CustomMetric(Metric):

  @dataclass
  class Config(Metric.Config):
    model_name_or_path: str = None
    nli_model_name: str = None
    embedding_module_name: str = "embeddings"
    encoder_module_name: str = "encoder"
    classifier_module_name: str = "classifier"
    alignment_type: str = "sinkhorn_stabilized_unbalanced"
    alignment_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

  def __init__(self, cfg):
    super().__init__(cfg)

    self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    self.model = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path).to(self.device)
    self.embedder = getattr(self.model, cfg.embedding_module_name)
    self.unembedder = UnEmbedder(self.embedder, self.tokenizer.vocab_size, self.model.config.max_position_embeddings).to(self.device)
    self.encoder = getattr(self.model, cfg.encoder_module_name)
    self.classifier = getattr(self.model, cfg.classifier_module_name).to("cpu")
    self.alignment = getattr(otu, cfg.alignment_type)
    self.alignment_kwargs = cfg.alignment_kwargs if cfg.alignment_kwargs else {"reg": 1.0, "reg_m": (0.5, 2.0)}
    self.nli_model = QKScore(tokenizer = self.tokenizer, model = self.model, device = self.device)

  def score(
        self, hypothesis: str, references: Tuple[torch.Tensor, torch.Tensor], source: Optional[str] = None
    ) -> float:
        """Calculate the score of the given hypothesis.
        """
        references = [torch.squeeze(ref) for ref in references]
        # shape of each reference tensor is (max_entity_length, embedding_size)

        tokenized_source_and_hp = self.tokenizer([source, hypothesis], padding = True, add_special_tokens = False, return_tensors = "pt")
        with torch.inference_mode():
          embedded_source_and_hp = self.embedder(tokenized_source_and_hp.input_ids.to(self.device))
          embedding_size = embedded_source_and_hp.shape[-1]

          embedded_triple = torch.cat([references[0], embedded_source_and_hp[1][:tokenized_source_and_hp.attention_mask[1].sum()], references[1]], dim = 0)
          mask_embedding = self.embedder(torch.tensor([self.tokenizer.mask_token_id]).to(tokenized_source_and_hp.input_ids.device)).reshape(1, embedding_size)
          embedded_masked_tail = torch.cat([references[0], embedded_source_and_hp[1][:tokenized_source_and_hp.attention_mask[1].sum()], torch.ones_like(references[1]) * mask_embedding], dim = 0)
          embedded_source = embedded_source_and_hp[0]

          padded_input = torch.nn.utils.rnn.pad_sequence([embedded_triple, embedded_source, embedded_masked_tail], batch_first = True).to(self.device)
          output = self.encoder(padded_input).last_hidden_state.cpu()

        # Now I want to align the first and the second row in a wise manner
        # Alignment 1: source <-> (ref1, hp, ref2)
        # Alignment 2: (ref1, hp, [MASK]) -> (ref1, hp, ref2)
        # Alignment 3: (ref1, hp, ref2) -> not (ref2, hp, ref1)
        pair_dists = torch.cdist(
            output[0],
            output[1]
        )
        pair_dists = torch.nn.functional.softmax(pair_dists, dim = 1)
        costs = self.alignment(
            a = torch.where(padded_input[0].sum(dim = 1) > 0, 1, 0).numpy(), 
            b = torch.where(padded_input[1].sum(dim = 1) > 0, 1, 0).numpy(), 
            M = pair_dists, 
            **self.alignment_kwargs
        )
        A = costs.argmax(dim = 1).unsqueeze(1)
        signs = torch.sign(A[-1] - A[0])
        moverscore = signs * torch.gather(pair_dists, dim = 1, index = A).mean()

        P_mask = self.classifier(output[2]).softmax(dim = -1)
        P_triple = self.classifier(output[0]).argmax(dim = -1)
        D2 = torch.gather(
            P_mask,
            dim = -1,
            index = P_triple
        )
        gptscore = torch.logsumexp(D2)

        unembedded_references = [self.unembedder(ref) for ref in references]
        decoded_references = [self.tokenizer.decode(ref) for ref in unembedded_references]
        
        qkscore = self.nli_model([(ref1, hypothesis, ref2) for ref1, ref2 in zip(decoded_references[0], decoded_references[1])])

        # Save scores to a list of records
        self.scores.append({
            "moverscore": moverscore.item(),
            "gptscore": gptscore.item(),
            "qkscore": qkscore.item(),
            "references": decoded_references,
            "hypothesis": hypothesis,
            "source": source,
        })

        return torch.stack([moverscore, gptscore, qkscore]).mean(dim = 0)

  def batch_scores(
        self,
        hypothesis: str,
        references: List[Tuple[torch.Tensor, torch.Tensor]],
        sources: List[str],
        batch_size: int = 16
    ) -> torch.Tensor:
        """Batch version of score: efficiently processes a list of (hypothesis, references, source) triples.

        All tokenization and embedding steps are amortized over the full batch.
        The encoder is called once over all 3 × batch_size sequences.

        Args:
            hypothesis: Hypothesis string.
            references: Per-sample pair of reference tensors (ref1, ref2), each of
                shape (max_entity_length, embedding_size), list of length M.
            sources: List of source strings, of length M, i.e. one source for each reference pair.
            batch_size: Batch size for the encoder.

        Returns:
            Tensor of shape (batch_size,) with per-sample mean scores.
        """
        references = [(torch.squeeze(r[0]), torch.squeeze(r[1])) for r in references]

        # --- 1. Batch tokenize sources and hypotheses ---
        tokenized_sources = self.tokenizer(
            sources, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt"
        )
        tokenized_hypothesis = self.tokenizer(
            hypothesis, add_special_tokens=False, return_tensors="pt"
        )

        with torch.inference_mode():
            # --- 2. Batch embed ---
            embedded_sources = self.embedder(tokenized_sources.input_ids.to(self.device))
            embedded_hypotheses = self.embedder(tokenized_hypothesis.input_ids.to(self.device)).reshape(1, -1, embedded_sources.shape[-1])
            embedding_size = embedded_hypotheses.shape[-1]

            mask_embedding = self.embedder(
                torch.tensor([self.tokenizer.mask_token_id], device=self.device)
            ).reshape(1, embedding_size)

            # --- 3. Build per-sample variable-length sequences ---
            triples: List[torch.Tensor] = []
            sources_emb: List[torch.Tensor] = []
            masked_tails: List[torch.Tensor] = []

            hp_len = int(tokenized_hypothesis.attention_mask.sum())

            for i in range(batch_size):
                src_len = int(tokenized_sources.attention_mask[i].sum())
                ref1, ref2 = references[i]

                hp_emb = embedded_hypotheses[i, :hp_len]
                src_emb = embedded_sources[i, :src_len]

                triples.append(torch.cat([ref1, hp_emb, ref2], dim=0))
                sources_emb.append(src_emb)
                masked_tails.append(
                    torch.cat([ref1, hp_emb, torch.ones_like(ref2) * mask_embedding], dim=0)
                )

            # Interleave as [triple_0, src_0, masked_0, triple_1, src_1, masked_1, ...]
            interleaved: List[torch.Tensor] = []
            for i in range(batch_size):
                interleaved.extend([triples[i], sources_emb[i], masked_tails[i]])

            seq_lengths = [s.shape[0] for s in interleaved]

            # --- 4. Single encoder forward pass over all 3 * batch_size sequences ---
            padded = torch.nn.utils.rnn.pad_sequence(
                interleaved, batch_first=True
            ).to(self.device)
            # shape: (3 * batch_size, max_seq_len, embedding_size)

            output = self.encoder(padded).last_hidden_state.cpu()
            # shape: (3 * batch_size, max_seq_len, hidden_size)

        padded_cpu = padded.cpu()

        # --- 5. Per-sample score computation ---
        moverscore_list: List[torch.Tensor] = []
        gptscore_list: List[torch.Tensor] = []

        for i in range(batch_size):
            idx = i * 3
            len_triple = seq_lengths[idx]
            len_source = seq_lengths[idx + 1]
            len_masked = seq_lengths[idx + 2]

            out_triple = output[idx, :len_triple]
            out_source = output[idx + 1, :len_source]
            out_masked = output[idx + 2, :len_masked]

            pad_triple = padded_cpu[idx, :len_triple]
            pad_source = padded_cpu[idx + 1, :len_source]

            # MoverScore: optimal-transport alignment of triple vs. source
            pair_dists = torch.cdist(out_triple, out_source)
            pair_dists = torch.nn.functional.softmax(pair_dists, dim=1)
            costs = self.alignment(
                a=torch.where(pad_triple.sum(dim=1) > 0, 1, 0).numpy(),
                b=torch.where(pad_source.sum(dim=1) > 0, 1, 0).numpy(),
                M=pair_dists,
                **self.alignment_kwargs,
            )
            A = costs.argmax(dim=1).unsqueeze(1)
            signs = torch.sign(A[-1] - A[0])
            moverscore_list.append(signs * torch.gather(pair_dists, dim=1, index=A).mean())

            # GPTScore: log-likelihood of the masked tail under the triple's predictions
            P_mask = self.classifier(out_masked).softmax(dim=-1)
            P_triple_idx = self.classifier(out_triple).argmax(dim=-1)
            D2 = torch.gather(P_mask, dim=-1, index=P_triple_idx.unsqueeze(-1)).squeeze(-1)
            gptscore_list.append(torch.logsumexp(D2, dim=0))

        moverscore_tensor = torch.stack(moverscore_list)
        gptscore_tensor = torch.stack(gptscore_list)

        # --- 6. QKScore (already handles a batch of triples) ---
        unembedded_refs = [
            (self.unembedder(references[i][0]), self.unembedder(references[i][1]))
            for i in range(batch_size)
        ]
        decoded_refs = [
            (self.tokenizer.decode(ur[0]), self.tokenizer.decode(ur[1]))
            for ur in unembedded_refs
        ]
        qkscore_tensor = self.nli_model(
            [(decoded_refs[i][0], hypothesis, decoded_refs[i][1]) for i in range(batch_size)]
        )

        # --- 7. Persist per-sample records ---
        for i in range(batch_size):
            self.scores.append({
                "moverscore": moverscore_list[i].item(),
                "gptscore": gptscore_list[i].item(),
                "qkscore": qkscore_tensor[i].item(),
                "references": list(decoded_refs[i]),
                "hypothesis": hypothesis,
                "source": sources[i],
            })

        return torch.stack([moverscore_tensor, gptscore_tensor, qkscore_tensor]).mean(dim=0)


class BatchMatrixFactorizationALS(MatrixFactorizationALS):

    def batch_factorize(self,
        matrices: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        niter: int = 30,
        tolerance: float = 1e-4,
        seed: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factorize the given matrix.

        Args:
            matrices (Tensor): Input matrices, shape `(N, M, R)`.
            observed_mask (Tensor, optional): Boolean mask of valid indices of shape `(N, M, R)`.
            niter (int): The number of alternating steps performed.
            tolerance (float): If the difference between the previous and current loss
              is smaller this value, ALS is regarded as converged.
            seed (int): A seed for the random number generator.

        Returns:
            Tensor: Low-rank matrices `X` of shape `(N, M, r)`.
            Tensor: Low-rank matrices `Y` of shape `(N, M, r)`.
        """
        rng = torch.Generator(matrices.device)
        rng = rng.manual_seed(seed)

        N, M, R = matrices.size()
        # Initialization:
        # Empirically observed the convergence to be much better with the scaled initialization.
        X = (
            torch.rand((N, M, self.rank), generator=rng, device=matrices.device)
            * (N * self.rank) ** -0.5
        )
        Y = (
            torch.rand((N, M, self.rank), generator=rng, device=matrices.device)
            * (M * self.rank) ** -0.5
        )
        if observed_mask is None:
            observed_mask = matrices.new_ones((N, M, R), dtype=torch.bool)

        regularization_term = self.regularization_weight * torch.eye(
            self.rank, device=matrices.device
        ).unsqueeze(0)

        prev_loss = float("1e5")
        for _ in range(niter):
            with timer.measure("ALS/iteration"):
                # A: r x r
                # B: r x N
                # Solve Ax = b
                Y_T = Y.transpose(1, 2)
                X_T = X.transpose(1, 2)
                observed_mask_T = observed_mask.transpose(1, 2)
                X = torch.linalg.solve(
                    Y_T[:, None, :, :] @ (Y[:, None, :, :] * observed_mask[:, :, :, None])
                    + regularization_term,
                    matrices @ Y,
                )
                Y = torch.linalg.solve(
                    X_T[:, None, :, :] @ (X[:, None, :, :] * observed_mask_T[:, :, :, None])
                    + regularization_term,
                    matrices.transpose(1, 2) @ X,
                )
                loss = self.compute_loss(matrices, X, Y, observed_mask=observed_mask)
                if prev_loss - loss <= tolerance:
                    break
                prev_loss = loss
        return X, Y


class BatchProbabilisticMBR(DecoderProbabilisticMBR):
    """Batch version of the probabilistic MBR algorithm, with one list of H hypotheses, N sources, and N lists of references, where N is the number of entity pairs."""

    def triwise_scores_probabilistic(
        self,
        hypotheses: List[str],
        references: List[List[torch.Tensor | str]],
        sources: List[str],
        batch_size: int = 16
    ) -> torch.Tensor:
        """Compute approximated pairwise scores using the probabilistic MBR algorithm.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            Tensor: Approximated pairwise scores of shape `(H, R)`.
        """
        assert len(references) == len(sources)
        rng = torch.Generator().manual_seed(self.cfg.seed)
        H = len(hypotheses)
        R = max(map(len, references))
        N = len(sources)
        num_ucalcs = math.ceil(H * R / self.cfg.reduction_factor)

        triwise_scores = torch.zeros((N, H, R), device=self.metric.device)
        # Choose the indices of hypotheses and references to sample for each source
        pairwise_sample_indices = torch.randperm(H * R, generator=rng)[:num_ucalcs]
        hypothesis_sample_indices: list[int] = (pairwise_sample_indices // R).tolist()
        reference_sample_indices: list[int] = (pairwise_sample_indices % R).tolist()

        # Algorithm 2 in the paper.
        if isinstance(self.metric, MetricCacheable):
            with timer.measure("encode/hypotheses"):
                hypotheses_ir = self.metric.encode(hypotheses)
            if hypotheses == references:
                references_ir = hypotheses_ir
            else:
                with timer.measure("encode/references"):
                    references_ir = self.metric.encode(references)
            if sources is None:
                source_ir = None
            else:
                with timer.measure("encode/source"):
                    source_ir = self.metric.encode([sources])

            num_hyp_samples = len(hypothesis_sample_indices)
            for i in range(0, num_hyp_samples, H):
                for n in range(0, N, batch_size):
                    triwise_scores[
                        n : n + batch_size,
                        hypothesis_sample_indices[i : i + H],
                        reference_sample_indices[i : i + H]
                    ] = self.metric.batch_scores(
                        hypotheses_ir[hypothesis_sample_indices[i : i + H]],
                        references_ir[reference_sample_indices[i : i + H]],
                        sources[n : n + batch_size]
                        if sources is not None
                        else None,
                    ).float()
        else:
            hypothesis_samples = [hypotheses[i] for i in hypothesis_sample_indices]
            
            for n in range(0, N, batch_size):
                reference_samples = [[references[m][j] for j in reference_sample_indices] for m in range(n, n + batch_size)]
                triwise_scores[n : n + batch_size, hypothesis_sample_indices, reference_sample_indices] = (
                    self.metric.batch_scores(
                        hypothesis_samples,
                        reference_samples,
                        sources[n : n + batch_size]
                        if sources is not None
                        else None,
                    ).float()
                )
        observed_mask = triwise_scores.new_zeros((N, H, R), dtype=torch.bool)
        observed_mask[:, hypothesis_sample_indices, reference_sample_indices] = True

        # Algorithm 1 in the paper.
        als = BatchMatrixFactorizationALS(
            regularization_weight=self.cfg.regularization_weight, rank=self.cfg.rank
        )
        X, Y = als.factorize(
            triwise_scores,
            observed_mask=observed_mask,
            niter=self.cfg.niter,
            seed=self.cfg.seed,
        )
        reconstructed_triwise_scores = X @ Y.T
        # reconstruct

        return reconstructed_triwise_scores

    def batch_decode(
        self,
        hypotheses: list[str],
        references: list[List[torch.Tensor | str]],
        sources: list[str],
        nbest: int = 5,
        reference_lprobs: Optional[torch.Tensor] = None,
        batch_size: int = 16,
    ) -> DecoderMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderMBR.Output: The n-best hypotheses.
        """

        if self.cfg.reduction_factor <= 1.0:
            expected_scores = self.metric.expected_scores(
                hypotheses, references, sources, reference_lprobs=reference_lprobs
            )
        else:  # Probabilistic MBR decoding
            triwise_scores_probabilistic = self.triwise_scores_probabilistic(
                hypotheses, references, sources, batch_size=batch_size
            )
            expected_scores = functional.expectation(
                triwise_scores_probabilistic, lprobs=reference_lprobs
            )

        selector_outputs = self.select(
            hypotheses, expected_scores, nbest=nbest, source=sources
        )

        return (
            self.Output(
                idx=selector_outputs.idx,
                sentence=selector_outputs.sentence,
                score=selector_outputs.score,
            )
            | selector_outputs
        )


class RelationDisambiguation():

    def __init__(self, model_name: str, hypotheses: List[str], topk: int = 5, batch_size: int = 16):

        self.model_name = model_name
        self.topk = topk
        self.batch_size = batch_size

        metric_cfg = CustomMetric.Config(
            model_name=model_name
        )
        self.metric = CustomMetric(metric_cfg)
        decoder_cfg = DecoderProbabilisticMBR.Config()
        self.decoder = DecoderProbabilisticMBR(decoder_cfg, self.metric)

        self.hypotheses = hypotheses
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, input_embeddings: List[List[torch.Tensor]], sources: List[str], show_progress_bar: bool = True) -> torch.Tensor:

        assert len(input_embeddings) == len(sources)

        references = [[(L[i], L[j]) for i in range(len(L)) for j in range(len(L)) if i != j] for L in input_embeddings]
    
        pseudorefs = tqdm(references, desc = "Generating pseudoreferences", disable = not show_progress_bar)
        candidates = [self.tokenizer.special_tokens_map['mask_token'] + " " + p + " " + self.tokenizer.special_tokens_map['mask_token'] for p in self.hypotheses]

        output = self.decoder.batch_decode(candidates, pseudorefs, sources, nbest=self.topk, batch_size=self.topkbatch_size)

        return output