"""Functions to compute baseline scores, i.e. mean integrated directional gradients and Shapley Interaction Quantification."""

from typing import List, Optional, Sequence, Tuple, Union
import gc
import os

import numpy as np
from shapiq import Game, KernelSHAPIQ
from shapiq.interaction_values import InteractionValues
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from mbeer.pullback import (
    OutputOnlyModel,
    compute_pred_ids_and_eq_class_emb_ids,
    jacobian,
)


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


@torch.no_grad()
def _preprocess_batch(
    batch,
    tokenizer,
    model,
    entity_span_combinations,
    max_tokens_per_entity,
    M,
    minibatch_size,
    num_workers,
    encoder_name="encoder",
):

    model_config = (
        model.model.config if isinstance(model, OutputOnlyModel) else model.config
    )

    tokenized_batch = tokenizer(
        batch,
        padding="max_length",
        add_special_tokens=False,
        max_length=(
            model_config.max_position_embeddings - 2
            if hasattr(model_config, "max_position_embeddings")
            else model_config.n_ctx - 2
        ),
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    batch_size = tokenized_batch.input_ids.shape[0]
    max_len = tokenized_batch.input_ids.shape[1]

    if isinstance(model, OutputOnlyModel):
        embedder = (
            model.model.base_model.embeddings
            if hasattr(model.model.base_model, "embeddings")
            else model.model.base_model.embed_tokens
        )
        output_only_model = model
    else:
        embedder = (
            model.base_model.embeddings
            if hasattr(model.base_model, "embeddings")
            else model.base_model.embed_tokens
        )
        output_only_model = OutputOnlyModel(model, encoder_name=encoder_name)

    embedded_batch = embedder(tokenized_batch.input_ids).unsqueeze(
        1
    )  # shape = (batch_size, 1, max_len, embedding_size)
    embedding_size = embedded_batch.shape[-1]

    # m_ratios should apply only to the entity spans
    entity_spans = [
        list(set(sum([list(combo) for combo in combo_list], [])))
        for combo_list in entity_span_combinations
    ]

    pred_ids, eq_class_emb_ids = compute_pred_ids_and_eq_class_emb_ids(
        entity_span_combinations, entity_spans, max_tokens_per_entity
    )

    #### PRED_IDS has one record per entity span and one per entity span combination, and the same will apply to EQ_CLASS_EMB_IDS
    repetitions = torch.as_tensor([len(ent_span_list) + len(span_combo_list) for ent_span_list, span_combo_list in zip(entity_spans, entity_span_combinations)])
    extended_embedded_batch = torch.repeat_interleave(embedded_batch, repetitions.to(embedded_batch.device), dim=0)
    extended_attention_mask = torch.repeat_interleave(tokenized_batch.attention_mask, repetitions.to(tokenized_batch.attention_mask.device), dim=0)

    m_ratios = (
        torch.linspace(0, 1, M + 1).reshape(1, -1, 1, 1).tile(extended_embedded_batch.shape[0], 1, max_len, 1)
    )  # shape = (1, M+1)
    for i, span_list in enumerate(entity_spans):
        for span in span_list:
            m_ratios[i, :, span[0] : span[1]] = 1.0
            
    extended_embedded_batch = (extended_embedded_batch * m_ratios.to(extended_embedded_batch.device)).reshape(-1, max_len, embedding_size).to("cpu")
    extended_attention_mask = extended_attention_mask.repeat_interleave(M + 1, dim=0).to("cpu")
    extended_pred_ids = pred_ids.repeat_interleave(M + 1, dim=0).to("cpu")

    local_dataset = TensorDataset(extended_embedded_batch, extended_attention_mask, extended_pred_ids)

    return {
        "output_only_model": output_only_model,
        "local_dataset": local_dataset,
        "pred_ids": extended_pred_ids,
        "eq_class_emb_ids": eq_class_emb_ids,
        "embedded_batch": embedded_batch,
        "extended_embedded_batch": extended_embedded_batch,
        "entity_span_combinations": entity_span_combinations,
        "entity_spans": entity_spans,
        "batch_size": batch_size,
        "max_len": max_len,
        "embedding_size": embedding_size,
    }


def mean_integrated_directional_gradients(
    model: AutoModelForMaskedLM,
    batch: List[str],
    tokenizer: AutoTokenizer,
    entity_span_combinations: List[List[Tuple[int, int]]],
    max_tokens_per_entity: int = 8,
    M: int = 10,
    minibatch_size: int = 4,
    num_workers: int = 2,
    encoder_name="encoder",
):
    """
    Compute the mean integrated directional gradients for a masked language model.
    Args:
        model: A masked language model.
        batch: A list of strings.
        tokenizer: A tokenizer.
        entity_span_combinations: A list of lists of tuples, each representing an entity span combination.
        max_tokens_per_entity: The maximum number of tokens per entity span.
        M: The number of integration steps.
        minibatch_size: The size of the minibatch.
        num_workers: The number of workers to use.

    Returns:
        A list of scores for each entity span combination.
    """

    batch_dict = _preprocess_batch(
        batch,
        tokenizer,
        model,
        entity_span_combinations,
        max_tokens_per_entity,
        M,
        minibatch_size,
        num_workers,
        encoder_name=encoder_name,
    )
    local_dataloader = DataLoader(batch_dict["local_dataset"], batch_size=minibatch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    output_only_model = batch_dict["output_only_model"]
    output_only_model.eval()
    output_only_model.requires_grad_(False)

    pred_ids = batch_dict["pred_ids"]
    eq_class_emb_ids = batch_dict["eq_class_emb_ids"]
    embedded_batch = batch_dict["embedded_batch"]
    batch_size = embedded_batch.shape[0]
    embedding_size = embedded_batch.shape[-1]
    entity_span_combinations = batch_dict["entity_span_combinations"]
    entity_spans = batch_dict["entity_spans"]
    gradients = []
    torch.cuda.empty_cache()
    gc.collect()

    for minibatch in tqdm(
        local_dataloader, desc="Mean Integrated Directional Gradients"
    ):
        minibatch_inputs, minibatch_attn_mask, minibatch_pred_ids = minibatch
        minibatch_inputs = minibatch_inputs.to(output_only_model.device)
        minibatch_attn_mask = minibatch_attn_mask.to(output_only_model.device)
        minibatch_pred_ids = minibatch_pred_ids.to(output_only_model.device)

        truncation_mask = minibatch_attn_mask.any(dim = 0).squeeze()
        minibatch_inputs = minibatch_inputs[:, truncation_mask]
        minibatch_attn_mask = minibatch_attn_mask[:, truncation_mask]
        pred_ids_truncation_mask = torch.any(minibatch_pred_ids >= 0, dim = 0).squeeze()
        minibatch_pred_ids = minibatch_pred_ids[:, pred_ids_truncation_mask]

        jac, preds = jacobian(
            minibatch_inputs,
            output_only_model,
            select="max",
            pred_id=minibatch_pred_ids,
            attention_mask=minibatch_attn_mask,
        )
        # jac.shape = (minibatch_size, max_len_pred_ids, 1, input_seq_len, embedding_size)
        # preds.shape = (minibatch_size, max_len_pred_ids, 1)
        jac = jac.to("cpu")
        del preds
        gradients.extend([J for J in jac.squeeze(2)])
        torch.cuda.empty_cache()
        gc.collect()

    output_only_model = output_only_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    eq_class_emb_ids = [L * (M + 1) for L in eq_class_emb_ids]
    max_pred_ids = max(map(len, pred_ids))

    J = torch.nn.utils.rnn.pad_sequence(
        [T[L[L < T.shape[0]]] for i, L in enumerate(eq_class_emb_ids) for T in gradients[i].reshape(-1, gradients[i].shape[-2], embedding_size)],
        batch_first=True,
        padding_value=0,
    ).reshape(batch_size, max_pred_ids, -1, embedding_size)

    X = torch.nn.utils.rnn.pad_sequence(
        [T for i, L in enumerate(eq_class_emb_ids) for T in embedded_batch[i, :, L, :]],
        batch_first=True,
        padding_value=0,
    ).reshape(batch_size, 1, -1, embedding_size)

    grad_dot_products = torch.sum(X * J, dim=-1)
    chunks = torch.chunk(grad_dot_products, M + 1, dim=0)
    mean_integrated_gradients = []
    for chunk, combo_list, span_list in zip(
        chunks, entity_span_combinations, entity_spans
    ):
        idg = chunk.mean(dim=0).flatten()
        span_map = {
            span: slice(
                sum(end - start for start, end in span_list[:i]),
                sum(end - start for start, end in span_list[: i + 1]),
            )
            for i, span in enumerate(span_list)
        }
        for combo in combo_list:
            mean_idg = torch.cat(
                [idg[span_map[combo[0]]], idg[span_map[combo[1]]]]
            ).mean()
            mean_integrated_gradients.append(mean_idg)

    return torch.stack(mean_integrated_gradients)


class MaskedLMShapleyGame(Game):
    """Shapley-Interaction-Quantification game for a masked language model.

    The model is encapsulated by :class:`mbeer.pullback.OutputOnlyModel` so the
    forward pass operates on input *embeddings* (the embedder is bypassed).
    Each *player* is an entity span (a token range) of a single input example;
    the value of a coalition ``S`` is the predicted probability of
    ``target_token_id`` at position ``target_pred_id`` after the embeddings of
    every entity span *not* in ``S`` are replaced with the mask-token
    embedding.

    This mirrors the ``SentimentClassificationGame`` example in the shapiq
    tutorial (where players are individual tokens and the value is the signed
    sentiment of the masked sentence) but masks at the *embedding* level and
    at the *span* level, which is the granularity used by the rest of
    ``mbeer``.

    Combine the game with any shapiq approximator (``KernelSHAPIQ``,
    ``SPEX``, ``ProxySPEX`` ...) to obtain Shapley Interaction Quantification
    indices (SII / k-SII / STII / FSII / ...). See
    :func:`shapley_interaction_quantification` for a one-call wrapper.

    Args:
        model: A masked-LM (or any classifier with a base-model + LM-head
            structure understood by ``OutputOnlyModel``).
        tokenizer: The matching tokenizer.
        text: The single input sentence to be explained.
        entity_spans: ``[(start, end), ...]`` token-index spans, one per
            player. Indices are relative to the tokenization with
            ``add_special_tokens=False`` (matching the rest of ``mbeer``).
        target_pred_id: Position(s) at which to read the prediction. Defaults
            to the first token of the first entity span. May be an int or a
            list of ints, in which case the value is averaged over positions.
        target_token_id: Vocab/class id(s) whose probability is the game
            value. Must match the length of ``target_pred_id``. If ``None``,
            the model's argmax prediction(s) at ``target_pred_id`` for the
            grand coalition are used.
        encoder_name: Name of the encoder submodule, forwarded to
            ``OutputOnlyModel``. Defaults to ``"encoder"``.
        minibatch_size: Number of coalitions evaluated per forward pass.
        device: Device on which to run the model. Defaults to the model's
            current device.
        normalize: If ``True`` (default) the game value is centered at the
            empty coalition (``v(\u2205) = 0``).
        verbose: Forwarded to :class:`shapiq.Game`.

    Examples:
        >>> from transformers import AutoTokenizer, AutoModelForMaskedLM
        >>> tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        >>> model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
        >>> text = "Paris is the capital of France"
        >>> spans = [(0, 1), (2, 4), (4, 6)]  # ["Paris", "the capital", "of France"]
        >>> game = MaskedLMShapleyGame(model, tokenizer, text, spans)
        >>> game.n_players
        3
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        text: List[str],
        entity_spans: List[Sequence[Tuple[int, int]]],
        *,
        target_pred_id: Optional[Union[int, Sequence[int]]] = None,
        target_token_id: Optional[Union[int, Sequence[int]]] = None,
        encoder_name: str = "encoder",
        minibatch_size: int = 16,
        device: Optional[Union[torch.device, str]] = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        if len(entity_spans) == 0:
            raise ValueError("`entity_spans` must contain at least one span (player).")

        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.entity_spans: List[Tuple[int, int]] = entity_spans
        self.encoder_name = encoder_name
        self.minibatch_size = int(minibatch_size)

        if device is None:
            device = next(model.parameters()).device
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        self.output_only_model = (
            OutputOnlyModel(model, encoder_name=encoder_name).to(self.device).eval()
        )

        max_len = (
            model.config.max_position_embeddings - 2
            if hasattr(model.config, "max_position_embeddings")
            else getattr(model.config, "n_ctx", 514) - 2
        )
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids: torch.Tensor = encoded.input_ids.to(self.device)  # (1, seq_len)
        # Encoder modules in `transformers` expect attention masks to be bool
        # or float (long is rejected by SDPA), but tokenizers return long.
        self.attention_mask: torch.Tensor = (
            encoded.attention_mask.to(self.device).bool().squeeze(0)  # (seq_len,)
        )
        self.seq_len = int(self.input_ids.shape[1])

        # `OutputOnlyModel` forwards the 2D attention mask straight into the
        # encoder, which (e.g. for BERT) expects an extended 4D mask. Since we
        # tokenize a single example without padding, the mask is all-ones and
        # we can simply pass ``None`` (let the encoder attend to everything).
        # Anything else would require the caller to expand the mask manually.
        self._forwarded_attention_mask: Optional[torch.Tensor] = (
            None if bool(self.attention_mask.all()) else self.attention_mask
        )

        for s, e in self.entity_spans:
            if not (0 <= s < e <= self.seq_len):
                raise ValueError(
                    f"Invalid entity span ({s}, {e}); must satisfy "
                    f"0 <= start < end <= {self.seq_len}."
                )

        embedder = (
            model.base_model.embeddings
            if hasattr(model.base_model, "embeddings")
            else model.base_model.embed_tokens
        )
        with torch.no_grad():
            self.input_embedding: torch.Tensor = embedder(
                self.input_ids
            )  # (1, seq_len, d)
            self._embedding_dim = int(self.input_embedding.shape[-1])
            self._embedding_dtype = self.input_embedding.dtype

            mask_token_id = getattr(tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                self.mask_embedding = torch.zeros(
                    self._embedding_dim, device=self.device, dtype=self._embedding_dtype
                )
            else:
                mask_input = torch.tensor([[int(mask_token_id)]], device=self.device)
                self.mask_embedding = embedder(mask_input).reshape(self._embedding_dim)

        self.target_pred_id: List[int] = self._coerce_int_list(
            target_pred_id, default=self.entity_spans, name="target_pred_id"
        )
        for p in self.target_pred_id:
            if not (0 <= p < self.seq_len):
                raise ValueError(
                    f"target_pred_id {p} out of range [0, {self.seq_len})."
                )

        if target_token_id is None:
            with torch.no_grad():
                logits, _ = self.output_only_model(
                    self.input_embedding,
                    pred_id=self.target_pred_id,
                    attention_mask=self._forwarded_attention_mask,
                )
                argmax = logits.argmax(dim=-1).reshape(-1).tolist()
            self.target_token_id: List[int] = [int(t) for t in argmax]
        else:
            self.target_token_id = self._coerce_int_list(
                target_token_id, default=self.entity_spans, name="target_token_id"
            )

        if len(self.target_token_id) != len(self.target_pred_id):
            raise ValueError(
                "`target_token_id` and `target_pred_id` must have the same length "
                f"(got {len(self.target_token_id)} vs {len(self.target_pred_id)})."
            )

        self._given_predictions = torch.tensor(
            self.target_token_id, device=self.device, dtype=torch.long
        )  # (1, n_targets)

        n_players = max(map(len, self.entity_spans))

        normalization_value: Optional[float] = None
        if normalize:
            empty = np.zeros((1, n_players), dtype=bool)
            normalization_value = float(self._evaluate_coalitions(empty)[0])

        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=normalization_value,
            player_names=entity_spans,
            verbose=verbose,
        )

    @staticmethod
    def _coerce_int_list(
        x: Optional[Union[int, Sequence[int]]],
        default: Optional[Sequence[int]],
        name: str,
    ) -> List[int]:
        if x is None:
            if default is None:
                raise ValueError(f"`{name}` is required.")
            if isinstance(default[0], (int, np.integer)):
                return [int(v) for v in default]
            elif isinstance(default[0], (tuple, list)):
                return [[int(v) for v in sublist] for sublist in default]
            else:
                raise TypeError(
                    f"`{name}` must be a sequence of ints or tuples of ints."
                )
        if isinstance(x, (int, np.integer)):
            return [int(x)]
        try:
            return [int(v) for v in x]
        except TypeError as exc:
            raise TypeError(f"`{name}` must be an int or a sequence of ints.") from exc

    @torch.no_grad()
    def _evaluate_coalitions(self, coalitions: np.ndarray) -> torch.Tensor:
        """Evaluate ``v(S)`` for each coalition row, before normalization."""
        coalitions = np.asarray(coalitions, dtype=bool)
        if coalitions.ndim == 1:
            coalitions = coalitions[None, :]
        if coalitions.shape[1] != len(self.entity_spans):
            raise ValueError(
                f"Got coalitions with {coalitions.shape[1]} columns but the game "
                f"has {len(self.entity_spans)} players."
            )
        n_coal = int(coalitions.shape[0])

        coalitions_t = torch.from_numpy(coalitions).to(
            self.device
        )  # (n_coal, n_players)
        embedding_mask = torch.zeros(
            n_coal, self.seq_len, dtype=torch.bool, device=self.device
        )
        for i, (s, e) in enumerate(self.entity_spans):
            embedding_mask[:, s:e] = embedding_mask[:, s:e] | (
                ~coalitions_t[:, i : i + 1]
            )

        # (n_coal, seq_len, d) — copy so the cached embeddings stay untouched.
        per_coal_emb = self.input_embedding.expand(n_coal, -1, -1).clone()
        per_coal_emb[embedding_mask] = self.mask_embedding

        gp_full = self._given_predictions.expand(n_coal, -1)  # (n_coal, n_targets)

        results: List[torch.Tensor] = []
        for start in range(0, n_coal, self.minibatch_size):
            end = start + self.minibatch_size
            batch_emb = per_coal_emb[start:end]
            batch_gp = gp_full[start:end]
            _, batch_proba = self.output_only_model(
                batch_emb,
                pred_id=torch.as_tensor(self.target_pred_id),
                attention_mask=self._forwarded_attention_mask,
                given_predictions=batch_gp,
            )
            # batch_proba has shape (batch, n_targets, 1); average across target positions.
            results.append(batch_proba.reshape(batch_emb.shape[0], -1).mean(dim=-1))

        return torch.cat(results, dim=0).to(torch.float32)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute the (un-normalized) game value of each coalition.

        ``shapiq.Game.__call__`` subtracts ``self.normalization_value`` after
        this method returns, so we deliberately do *not* center here.
        """
        return self._evaluate_coalitions(coalitions).cpu().numpy()


@torch.no_grad()
def shapley_interaction_quantification(
    model: torch.nn.Module,
    batch: List[str],
    tokenizer: AutoTokenizer,
    entity_spans: Sequence[Tuple[int, int]],
    *,
    max_order: int = 2,
    index: str = "k-SII",
    target_pred_id: Optional[Union[int, Sequence[int]]] = None,
    target_token_id: Optional[Union[int, Sequence[int]]] = None,
    budget: Optional[int] = None,
    minibatch_size: int = 16,
    encoder_name: str = "encoder",
    device: Optional[Union[torch.device, str]] = None,
    random_state: Optional[int] = None,
    normalize: bool = True,
    verbose: bool = False,
) -> Tuple[InteractionValues, MaskedLMShapleyGame]:
    """Compute Shapley Interaction Quantification indices for a masked LM.

    Builds a :class:`MaskedLMShapleyGame` over ``text``/``entity_spans`` and
    runs :class:`shapiq.KernelSHAPIQ` on it. Returns the interaction values
    together with the underlying game (so the caller can re-use it, e.g. with
    a different approximator or to inspect ``game.player_names``).

    Args:
        model, tokenizer, text, entity_spans: Forwarded to
            :class:`MaskedLMShapleyGame`.
        max_order: Highest interaction order to estimate (default ``2``).
        index: Interaction index. Defaults to ``"k-SII"``; see
            ``shapiq.KernelSHAPIQ`` for the full list.
        target_pred_id, target_token_id: Forwarded to the game.
        budget: Number of coalitions evaluated by the approximator. Defaults
            to ``min(2 ** n_players, 2048)`` which is exact for up to 11
            players and an approximation otherwise.
        minibatch_size: Coalitions per forward pass.
        encoder_name: Forwarded to :class:`OutputOnlyModel`.
        device: Forwarded to the game.
        random_state: Forwarded to the approximator.
        normalize: Forwarded to the game.
        verbose: Forwarded to the game.

    Returns:
        ``(interaction_values, game)`` where ``interaction_values`` is the
        :class:`shapiq.InteractionValues` returned by ``approximate``.
    """

    for text, sent_entity_spans in tqdm(
        zip(batch, entity_spans), desc="Shapley Interaction Quantification"
    ):

        game = MaskedLMShapleyGame(
            model=model,
            tokenizer=tokenizer,
            text=[text],
            entity_spans=[sent_entity_spans],
            target_pred_id=target_pred_id,
            target_token_id=target_token_id,
            encoder_name=encoder_name,
            minibatch_size=minibatch_size,
            device=device,
            normalize=normalize,
            verbose=verbose,
        )

        if budget is None:
            budget = min(2**game.n_players, 2048)

        approximator = KernelSHAPIQ(
            n=game.n_players,
            max_order=max_order,
            index=index,
            pairing_trick=True,
            random_state=random_state,
        )
        interaction_values = approximator.approximate(budget=budget, game=game)

        yield interaction_values.get_n_order_values(max_order)
