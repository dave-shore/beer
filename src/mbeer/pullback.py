from inspect import signature
from itertools import chain
import math
from typing import List, Tuple, Set
import warnings
import os
from functools import lru_cache
import torch
from torch.func import jacfwd, jacrev
from transformers import AutoModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
EPS = 2 * torch.finfo(torch.float32).eps


class OutputOnlyModel(torch.nn.Module):
    def __init__(self, model: AutoModel, encoder_name: str | List[str] = "encoder"):
        super().__init__()

        self.model = model
        self.model.eval()
        self.config = model.config
        self.base_model = model.base_model if hasattr(model, "base_model") else model

        self.hook_handles = []
        self.encoder_name = (
            encoder_name if isinstance(encoder_name, list) else [encoder_name]
        )
        self.device = (
            model.device
            if hasattr(model, "device")
            else next(model.parameters()).device
        )
        self.dtype = (
            model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype
        )
        self.encoder = torch.nn.ModuleList(
            [
                model.base_model.get_submodule(n)
                if hasattr(model, "base_model")
                else model.get_submodule(n)
                for n in self.encoder_name
            ]
        )
        assert len(self.encoder) == len(self.encoder_name)

        self.module_signatures = (
            {
                n: signature(module.forward)
                for n, module in self.model.base_model.named_children()
            }
            if hasattr(model, "base_model")
            else {n: signature(module.forward) for n, module in model.named_children()}
        )

        self.lm_head = torch.nn.Sequential(
            *[
                getattr(self.model, n, torch.nn.Identity())
                for n in [
                    n
                    for n, module in self.model.named_children()
                    if n
                    in [
                        "classifier",
                        "cls",
                        "lm_head",
                        "generator_lm_head",
                        "generator_predictions",
                    ]
                ]
            ]
        )

    @staticmethod
    def _max_hook(module, args, output):
        if hasattr(output, "logits"):
            output = output.logits
        return output.max(dim=-1, keepdim=True).values

    def forward(
        self,
        X: torch.Tensor,
        pred_id: List[int] = None,
        select: torch.Tensor = None,
        attention_mask=None,
        given_predictions: torch.Tensor = None,
        only_max_pred_id: bool = False,
    ):

        if only_max_pred_id:
            self.hook_handles.append(self.lm_head.register_forward_hook(self._max_hook))
        else:
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []

        if isinstance(pred_id, list):
            if len(pred_id) > 1 and isinstance(pred_id[0], list):
                pred_id = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(p) for p in pred_id],
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                pred_id = torch.as_tensor(pred_id)

        if hasattr(self.model, "base_model"):
            if X.dim() < 3:
                X = X.unsqueeze(0)

            batch_size, seq_len, embedding_size = X.shape
            if attention_mask is not None:
                if attention_mask.dim() == 1 or (
                    attention_mask.dim() == 2 and attention_mask.shape[1] != seq_len
                ):
                    attention_mask = attention_mask.reshape(1, -1).expand(
                        batch_size, -1
                    )
                elif attention_mask.shape[0] < batch_size:
                    attention_mask = attention_mask[[0] * batch_size].reshape(
                        batch_size, seq_len
                    )
                else:
                    attention_mask = attention_mask[:batch_size].reshape(
                        batch_size, seq_len
                    )

            for enc_module, enc_mod_name in zip(self.encoder, self.encoder_name):
                if "attention_mask" in self.module_signatures[enc_mod_name].parameters:
                    X = enc_module(X, attention_mask=attention_mask)
                elif "attn_mask" in self.module_signatures[enc_mod_name].parameters:
                    X = enc_module(X, attn_mask=attention_mask)
                elif "mask" in self.module_signatures[enc_mod_name].parameters:
                    X = enc_module(X, mask=attention_mask)
                else:
                    X = enc_module(X)

            encoder_output = (
                X.last_hidden_state if hasattr(X, "last_hidden_state") else X
            )

            encoder_output = (
                encoder_output[:, pred_id, :]
                if encoder_output.dim() == 3
                else encoder_output
            )
            y = self.lm_head(encoder_output)
            if hasattr(y, "logits"):
                y = y.logits

        else:
            encoder_output, _ = self.model.encoder(X)
            y = self.model.classifier(encoder_output[:, 0])
            if hasattr(y, "logits"):
                y = y.logits

        if select is not None:
            select = select.unsqueeze(0).expand(y.shape[0], -1, -1)
            select_on_pred_id = select[:, pred_id, :]
            y = torch.gather(y, dim=-1, index=select_on_pred_id)

        if only_max_pred_id:
            return y, torch.sigmoid(y)

        if given_predictions is not None:
            pred_proba = torch.gather(
                torch.softmax(y, dim=-1),
                dim=-1,
                index=(
                    given_predictions.unsqueeze(-1)
                    if given_predictions.dim() < y.dim()
                    else given_predictions
                ),
            )
            return y, pred_proba

        else:
            return y, torch.softmax(y, dim=-1)


@lru_cache(maxsize=128)
def jacobian(
    nn_input: torch.Tensor,
    model: torch.nn.Module = None,
    select: torch.Tensor | str | None = None,
    pred_id: List[int] = None,
    attention_mask=None,
    given_predictions: torch.Tensor = None,
    mode: str = "rev",
):
    """
    Computes the full Jacobian matrix of the neural network output with respect
    to its input.

    Args:
        nn_output (torch.Tensor): The output tensor of a neural network where
        each element depends on the input tensor and has gradients enabled.
        nn_input (torch.Tensor): The input tensor to the neural network with
        gradients enabled.

    Returns:
        torch.Tensor: A tensor representing the Jacobian matrix. The dimensions
        of the matrix will be [len(nn_output), len(nn_input)], reflecting the
        partial derivatives of each output element with respect to each input element.
    """

    if pred_id is None:
        pred_id = [0] * nn_input.shape[0]

    if select is None:
        select = [None] * nn_input.shape[0]

    # nn_input.shape = (batch_size, N_patches+1, embedding_size)
    if mode == "fwd":
        batch_jacobian_and_predictions = [
            jacfwd(model, argnums=0, has_aux=True)(
                x,
                pred_id[i],
                select[i] if isinstance(select, torch.Tensor) else None,
                None if attention_mask is None else attention_mask[i],
                None if given_predictions is None else given_predictions[i],
                (select == "max"),
            )
            for i, x in enumerate(nn_input.split(1))
        ]
    else:
        batch_jacobian_and_predictions = [
            jacrev(model, argnums=0, has_aux=True)(
                x,
                pred_id[i],
                select[i] if isinstance(select, torch.Tensor) else None,
                None if attention_mask is None else attention_mask[i],
                None if given_predictions is None else given_predictions[i],
                (select == "max"),
            )
            for i, x in enumerate(nn_input.split(1))
        ]

    nn_input.grad = None

    batch_jacobian = torch.nn.utils.rnn.pad_sequence(
        [J.squeeze(0) for J, _ in batch_jacobian_and_predictions],
        batch_first=True,
        padding_value=0,
    )
    predictions = torch.nn.utils.rnn.pad_sequence(
        [p.squeeze(0) for _, p in batch_jacobian_and_predictions],
        batch_first=True,
        padding_value=0,
    )
    # batch_jacobian.shape = (batch_size, output_size, (N_patches+1), embedding_size)

    batch_jacobian = torch.squeeze(
        batch_jacobian, dim=list(range(1, batch_jacobian.dim()))
    )
    predictions = torch.squeeze(predictions, dim=list(range(1, predictions.dim())))
    # batch_jacobian.shape = (batch_size, max_len_pred_ids, selected_vocab_size, input_seq_len, embedding_size)
    # predictions.shape = (batch_size, max_len_pred_ids, selected_vocab_size)

    return batch_jacobian, predictions


def chebyshev_filter_rank_estimate(
    matrix: torch.Tensor,
    eigenvalues: torch.Tensor,
    number_of_sample_vectors: int,
    degree: int,
) -> float:

    random_sample_vectors = torch.randn(
        number_of_sample_vectors,
        eigenvalues.shape[-1],
        device=eigenvalues.device,
        dtype=eigenvalues.dtype,
    )
    random_sample_vectors = random_sample_vectors / torch.linalg.norm(
        random_sample_vectors, dim=-1, keepdim=True
    )

    dos = torch.cos(eigenvalues) ** (-1)
    dos_diff = dos.diff(dim=-1)
    a = torch.min(eigenvalues[dos_diff > -EPS], dim=-1)[0]
    b = torch.max(eigenvalues, dim=-1)[0]
    gamma_coefficients = [
        eigenvalues.shape[-1] * (torch.cos(a) ** (-1) + torch.cos(b) ** (-1)) // math.pi
    ] + [
        eigenvalues.shape[-1]
        * (2 / math.pi)
        * (torch.sin(k * torch.cos(a) ** (-1)) + torch.sin(k * torch.cos(b) ** (-1)))
        / k
        for k in range(1, degree + 1)
    ]

    quadratic_forms = [
        torch.sum(
            (torch.special.chebyshev_polynomial_t(matrix, k) @ random_sample_vectors.T)
            * random_sample_vectors.T,
            dim=-2,
        ).mean(dim=-1)
        for k in range(degree + 1)
    ]

    r = torch.sum(torch.stack(quadratic_forms) * torch.stack(gamma_coefficients), dim=0)

    if torch.isnan(r).any():
        warnings.warn("NaN values in r")

    return r


@torch.no_grad()
def estimate_pdet(
    eigenvalues: torch.Tensor,
    det_regularization: float | str | None = "auto",
    min_rank: int | None = None,
) -> torch.Tensor:

    tr_rad_ratio = eigenvalues.sum(dim=-1) / (eigenvalues[..., 0] + EPS)
    tr_sq_rad_ratio = eigenvalues.pow(2).sum(dim=-1) / (
        eigenvalues.pow(2)[..., 0] + EPS
    )

    pade_approx = eigenvalues**2 / (
        eigenvalues[..., [0]] * (eigenvalues + 2 * eigenvalues[..., [0]]) + EPS
    )
    pade_approx = pade_approx.sum(dim=-1)
    d = eigenvalues.shape[-1]

    # Mid-point coefficient estimate for the pseudo-determinant
    rk_est = tr_sq_rad_ratio.reciprocal()
    rk_est = torch.clamp(
        rk_est, min=min_rank if min_rank is not None else 1, max=d
    ).unsqueeze(-1)

    # Smallest strictly-positive value of this dtype; used as a floor so that
    # `log(det_regularization)` never sees 0 (which yields -inf -> 0 * -inf = NaN).
    tiny = torch.finfo(eigenvalues.dtype).tiny

    if det_regularization == "auto":
        # For a fully-degenerate spectrum (e.g. a padded/empty slot) `rk_est`
        # saturates to `d`, making the exponent `1 / (d - rk_est)` blow up to
        # +inf and `eps ** inf == 0`. Clamp the denominator and the result so
        # the regularizer stays strictly positive.
        denom = (d - rk_est).clamp(min=1.0)
        det_regularization = (
            torch.finfo(eigenvalues.dtype).eps ** (1.0 / denom)
        ).clamp(min=tiny)
    elif det_regularization is None:
        det_regularization = torch.tensor(
            tiny, dtype=eigenvalues.dtype, device=eigenvalues.device
        ).unsqueeze(-1)
    else:
        det_regularization = (
            torch.tensor(
                det_regularization, dtype=eigenvalues.dtype, device=eigenvalues.device
            )
            .unsqueeze(-1)
            .clamp(min=tiny)
        )

    log_lb = torch.sum(
        torch.log(eigenvalues + det_regularization), dim=-1, keepdim=True
    ) - (d - rk_est) * torch.log(det_regularization)

    log_ub = (
        log_lb
        + det_regularization * tr_rad_ratio.unsqueeze(-1)
        + (1 - det_regularization) * pade_approx.unsqueeze(-1)
    )

    general_ub = (eigenvalues.sum(dim=-1, keepdim=True) / rk_est) ** rk_est

    lb = torch.exp(log_lb)
    lb = torch.clamp(
        lb,
        min=EPS * torch.ones_like(lb),
        max=torch.maximum(EPS * torch.ones_like(lb), general_ub),
    ).nan_to_num(nan=0.0)
    ub = torch.exp(log_ub)
    ub = torch.clamp(ub, min=lb, max=torch.maximum(lb, general_ub)).nan_to_num(nan=0.0)

    special_pdets = torch.where(torch.isfinite(ub), (ub + lb) / 2, lb)

    return special_pdets


@lru_cache(maxsize=128)
def _sample_span_positions(
    span: Tuple[int, int], max_tokens: int | None = None
) -> torch.Tensor:
    """Token-position indices to sample within `span`.

    Returns the full range when `width < max_tokens`; otherwise returns
    `max_tokens // 2` positions interleaved from each end (start, end-1,
    start+1, end-2, ...). Output shape is (k, 1), int64, matching the
    original ``arange(a, b)[idxs].reshape(-1, 1)`` contract.
    """

    if not isinstance(span, tuple):
        return torch.empty(0, 1, dtype=torch.int32)

    if len(span) == 1:
        return torch.tensor([span[0]], dtype=torch.int32).reshape(-1, 1)

    a, b = span
    if max_tokens is None or b - a < max_tokens:
        indices = torch.arange(a, b, dtype=torch.int32).reshape(-1, 1)
    else:
        indices = (
            torch.linspace(a, b - 1, max_tokens)
            .round()
            .to(dtype=torch.int32)
            .reshape(-1, 1)
        )

    return indices


def compute_pred_ids_and_eq_class_emb_ids(
    span_combinations: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    ent_spans: List[List[Tuple[int, int]]],
    max_tokens_per_entity: int,
):

    def _cached_sample(
        spans: List[Tuple[int, int]] | Set[Tuple[int, int]],
        take_out: Tuple[int, int] = None,
    ) -> torch.Tensor:

        if isinstance(spans, tuple) and len(spans) <= 2:
            spans = [spans]

        if len(spans) == 0:
            warnings.warn("No spans to sample")
            return torch.empty(0, 1, dtype=torch.int32)

        output = torch.empty(0, 1, dtype=torch.int32)
        for span in spans:
            s = _sample_span_positions(span, max_tokens_per_entity)
            output = torch.cat([output, s])

        if take_out is not None:
            if len(take_out) == 1:
                take_out = (take_out[0], take_out[0] + 1)
            output = output[
                torch.logical_or(output < take_out[0], output >= take_out[1])
            ]

        return output.reshape(-1, 1)

    eq_class_emb_ids = list(
        chain(
            *[
                (
                    [
                        _cached_sample(ent_span_list, take_out=ent)
                        for ent in ent_span_list
                    ]
                    + [_cached_sample([s, o]) for s, o in span_combo_list]
                )
                for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)
            ]
        )
    )
    eq_class_emb_ids = torch.nn.utils.rnn.pad_sequence(
        eq_class_emb_ids, batch_first=True, padding_value=-1
    ).squeeze(-1)
    # eq_class_emb_ids.shape = (sum(N_entities_per_document + N_pairs_per_document), max_perturbed_entity_spans_length)

    pred_ids = list(
        chain(
            *[
                (
                    [_cached_sample([ent]) for ent in ent_span_list]
                    + [_cached_sample([s, o]) for s, o in span_combo_list]
                )
                for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)
            ]
        )
    )
    pred_ids = torch.nn.utils.rnn.pad_sequence(
        pred_ids, batch_first=True, padding_value=-1
    ).squeeze(-1)
    # pred_ids.shape = (sum(N_entities_per_document + N_pairs_per_document), max_masked_entity_spans_length)

    return pred_ids, eq_class_emb_ids


def pullback(
    input_simec: torch.Tensor,
    g: torch.Tensor,
    model: torch.nn.Module = None,
    eq_class_emb_ids: List[List[int]] = None,
    select: torch.Tensor = None,
    pred_id: List[int] = None,
    degrowth: bool = True,
    same_equivalence_class: bool = True,
    attention_mask=None,
    return_trace: bool = False,
    return_predictions: bool = False,
    given_predictions: torch.Tensor = None,
    min_rank: int | None = None,
    min_num_eigenvectors: int = 32,
    max_num_eigenvectors: int = None,
    approximated_eigendecomposition: bool = False,
    det_regularization: float | str | None = "auto",
):
    """
    Computes the pullback metric tensor using the given input and output embeddings and a metric tensor g.

    Args:
        input_simec (torch.Tensor): Input embeddings tensor.
        output_simec (torch.Tensor): Output embeddings tensor derived from the
        input embeddings.
        g (torch.Tensor): Metric tensor g used as the Riemannian metric in the
        output space, of size (batch_size, embedding_size, embedding_size).
        eq_class_emb_ids (List[int], optional): Indices of embeddings to be
        considered for the pullback. If provided, restricts the computation to
        these embeddings.
        model (torch.nn.Module): Model to compute the Jacobian of.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Eigenvalues and eigenvectors of the
        pullback metric tensor.
    """
    if approximated_eigendecomposition:
        from cupyx.scipy.sparse.linalg import eigsh

    if min_rank is None and min_num_eigenvectors is not None:
        min_rank = min_num_eigenvectors
    elif min_rank is not None and min_num_eigenvectors is None:
        min_num_eigenvectors = min_rank
    elif min_rank is None and min_num_eigenvectors is None:
        min_rank = 1
        min_num_eigenvectors = 1

    input_simec = input_simec.to(dtype=torch.get_default_dtype())
    g = g.to(dtype=torch.get_default_dtype())
    if given_predictions is not None:
        given_predictions = given_predictions.to(dtype=torch.get_default_dtype())
    if model is not None:
        model = model.to(dtype=torch.get_default_dtype())

    if pred_id is None:
        pred_id = [0] * input_simec.shape[0]

    pred_id = tuple(tuple(p) for p in pred_id)
    batch_size = input_simec.shape[0]

    jac, predictions = jacobian(
        input_simec,
        model,
        select=select,
        pred_id=pred_id,
        attention_mask=attention_mask,
        given_predictions=given_predictions,
    )
    # jac.shape = (batch_size, max_len_pred_ids, selected_vocab_size, input_seq_len, embedding_size)
    # predictions.shape = (batch_size, max_len_pred_ids, selected_vocab_size)
    while jac.dim() < 5:
        jac = torch.unsqueeze(jac, 0)

    max_pred_ids = max(map(len, pred_id))

    jac = jac.reshape(
        input_simec.shape[0],
        max_pred_ids,
        select.shape[-1] if select is not None else 1,
        input_simec.shape[1],
        input_simec.shape[-1],
    )

    if select is not None:
        jac = torch.transpose(jac, 2, 3)
        # jac.shape = (batch_size, max_len_pred_ids, input_seq_len, selected_vocab_size, embedding_size)

    # Select ids and pad if necessary
    max_len = max(map(len, eq_class_emb_ids))
    max_idx = max(max(L) if L else -1 for L in eq_class_emb_ids)
    embedding_size = jac.shape[-1]
    assert max_idx < jac.shape[2], (
        f"eq_class_emb_ids index {max_idx} >= jac dim 2 ({jac.shape[2]})"
    )

    jac = torch.nn.utils.rnn.pad_sequence(
        [T for i, L in enumerate(eq_class_emb_ids) for T in jac[i, :, L, :]],
        batch_first=True,
        padding_value=0,
    ).reshape(batch_size, max_pred_ids, max_len, *jac.shape[-2:])
    # jac.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, selected_vocab_size, embedding_size)
    (
        batch_size,
        max_len_pred_ids,
        max_eq_class_emb_ids_length,
        selected_vocab_size,
        embedding_size,
    ) = jac.shape

    if max_num_eigenvectors is None:
        max_num_eigenvectors = jac.shape[-1] - 1

    #########################################################
    jac = (
        torch.stack(
            [
                torch.cat(
                    [chunk, chunk[-1:]], dim=0
                )  # take Jacobians 3-by-3 and duplicate the last row of each triplet
                for chunk in torch.split(jac, 3, dim=0)
            ]
        )
        .transpose(1, 3)
        .flatten(start_dim=3, end_dim=4)
    )
    # J_grouped.shape = (batch_size, max_eq_class_emb_ids_length, max_len_pred_ids, 4*selected_vocab_size, embedding_size)

    # To recover the eq_class_emb_ids of the independent variables in a pair, we can just take the pred_ids of the other element of each triplet. i.e. if positions I are dependent in row 0, they will be the independent positions in row 1.
    # Finally full pair eq_class_emb_ids are kept as is
    idx_indep_I = pred_id[1::3]
    idx_indep_J = pred_id[::3]
    idx_indep_pair = pred_id[2::3]

    if len(eq_class_emb_ids[::3]) > 0 and any(
        len(y) > 0 for y in eq_class_emb_ids[::3]
    ):
        if all(
            x[0] in y and x[-1] in y for x, y in zip(idx_indep_I, eq_class_emb_ids[::3])
        ):
            sep_indep_I = [
                (y.index(x[0]), y.index(x[-1]))
                for x, y in zip(idx_indep_I, eq_class_emb_ids[::3])
            ]
        else:
            warnings.warn(
                "Positions in pred_ids and eq_class_emb_ids do not match for I"
            )
            sep_indep_I = [(0, 1)]
    else:
        sep_indep_I = [(0, 1)]
    if len(eq_class_emb_ids[1::3]) > 0 and any(
        len(y) > 0 for y in eq_class_emb_ids[1::3]
    ):
        if all(
            x[0] in y and x[-1] in y
            for x, y in zip(idx_indep_J, eq_class_emb_ids[1::3])
        ):
            sep_indep_J = [
                (y.index(x[0]), y.index(x[-1]))
                for x, y in zip(idx_indep_J, eq_class_emb_ids[1::3])
            ]
        else:
            warnings.warn(
                "Positions in pred_ids and eq_class_emb_ids do not match for J"
            )
            sep_indep_J = [(0, 1)]
    else:
        sep_indep_J = [(0, 1)]

    sep_indep_pair = [
        (y.index(x[0]), y.index(x[-1])) for x, y in zip(idx_indep_I, idx_indep_pair)
    ]

    jac_I_num, jac_J_num, jac_I_den, jac_J_den = torch.chunk(jac, 4, dim=-2)

    jac_I_num = torch.nn.utils.rnn.pad_sequence(
        [
            jac_I_num.flatten(start_dim=2)[i, tup[0] : tup[1], :]
            for i, tup in enumerate(sep_indep_I)
        ],
        batch_first=True,
        padding_value=0,
    )
    jac_J_num = torch.nn.utils.rnn.pad_sequence(
        [
            jac_J_num.flatten(start_dim=2)[i, tup[0] : tup[1], :]
            for i, tup in enumerate(sep_indep_J)
        ],
        batch_first=True,
        padding_value=0,
    )
    jac_I_den = torch.nn.utils.rnn.pad_sequence(
        [
            jac_I_den.flatten(start_dim=2)[i, : tup[1], :]
            for i, tup in enumerate(sep_indep_pair)
        ],
        batch_first=True,
        padding_value=0,
    )
    jac_J_den = torch.nn.utils.rnn.pad_sequence(
        [
            jac_J_den.flatten(start_dim=2)[i, tup[1] :, :]
            for i, tup in enumerate(sep_indep_pair)
        ],
        batch_first=True,
        padding_value=0,
    )

    jac_I = torch.cat(
        [
            torch.nn.functional.pad(
                jac_I_num, (0, 0, 0, max_len_pred_ids - jac_I_num.shape[1])
            ).unflatten(
                dim=2, sizes=(max_len_pred_ids, selected_vocab_size, embedding_size)
            ),
            torch.nn.functional.pad(
                jac_I_den, (0, 0, 0, max_len_pred_ids - jac_I_den.shape[1])
            ).unflatten(
                dim=2, sizes=(max_len_pred_ids, selected_vocab_size, embedding_size)
            ),
        ],
        dim=-2,
    )
    jac_J = torch.cat(
        [
            torch.nn.functional.pad(
                jac_J_num, (0, 0, 0, max_len_pred_ids - jac_J_num.shape[1])
            ).unflatten(
                dim=2, sizes=(max_len_pred_ids, selected_vocab_size, embedding_size)
            ),
            torch.nn.functional.pad(
                jac_J_den, (0, 0, 0, max_len_pred_ids - jac_J_den.shape[1])
            ).unflatten(
                dim=2, sizes=(max_len_pred_ids, selected_vocab_size, embedding_size)
            ),
        ],
        dim=-2,
    )
    # jac_I.shape = (batch_size, max_eq_class_emb_ids_length, max_len_pred_ids, 2*selected_vocab_size, embedding_size)
    # jac_J.shape = (batch_size, max_eq_class_emb_ids_length, max_len_pred_ids, 2*selected_vocab_size, embedding_size)
    # From here on max_eq_class_emb_ids_length == max_len_pred_ids

    jac = torch.stack([jac_I, jac_J], dim=1).flatten(end_dim=1)
    # jac.shape = (batch_size*2, max_eq_class_emb_ids_length, max_len_pred_ids, 2*selected_vocab_size, embedding_size)
    g = g.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    with torch.no_grad():
        jac = torch.flatten(jac, end_dim=-3)
        # jac.shape = (batch_size*max_eq_class_emb_ids_length*max_len_pred_ids, selected_vocab_size, embedding_size)

        # g.shape = (batch_size, max_len, selected_vocab_size, selected_vocab_size)
        g_flat = torch.flatten(g, end_dim=1) if g.dim() == 4 else g
        g_flat = (
            g_flat[: jac.shape[0]]
            if g.shape[0] > jac.shape[0]
            else g_flat.tile(jac.shape[0] // g.shape[0] + 1, 1, 1)[: jac.shape[0]]
        )
        if jac.shape[-2] < g_flat.shape[-1]:
            jac = torch.nn.functional.pad(
                jac, (0, 0, 0, g_flat.shape[-1] - jac.shape[-2])
            )
        elif jac.shape[-1] > g_flat.shape[-1]:
            g_flat = torch.nn.functional.pad(
                g_flat, (0, jac.shape[-2] - g_flat.shape[-1], 0, 0)
            )

        if jac.shape[0] < g_flat.shape[0]:
            jac = torch.cat(
                [
                    jac,
                    torch.zeros(
                        g_flat.shape[0] - jac.shape[0],
                        *jac.shape[1:],
                        device=jac.device,
                        dtype=jac.dtype,
                    ),
                ]
            )

        jac_t = torch.transpose(jac, -2, -1)
        # jac_t.shape = (batch_size*max_len*max_len_pred_ids, embedding_size, selected_vocab_size)

        tmp = torch.bmm(jac_t, g_flat).to(dtype=torch.float64)
        # tmp.shape = (batch_size*max_len*max_len_pred_ids, embedding_size, selected_vocab_size)
        pullback_metric = torch.bmm(tmp, jac.to(dtype=torch.float64))
        # pullback_metric.shape = (batch_size*max_len*max_len_pred_ids, embedding_size, embedding_size)

        # Pairing i|j and j|i, then getting back to the original batch size.
        # `output_batch_size` holds the two conditional metrics per combination
        # in the order [i|j, j|i]. To restore the triplet layout used everywhere
        # downstream (row 0 = I|J, row 1 = J|I, row 2 = IJ|IJ), we duplicate the
        # *last* metric of each pair (j|i) into the third slot. The previous
        # `repeat_interleave(2)[:batch_size]` instead produced [i|j, i|j, j|i],
        # which put j|i at row 2 and a redundant copy of i|j at row 1.
        output_batch_size = batch_size // 3 * 2
        output_batch_size = max(2, output_batch_size)
        pullback_metric = (
            pullback_metric.unflatten(
                dim=0, sizes=(output_batch_size, max_len_pred_ids, max_len_pred_ids)
            )
            .transpose(1, 2)
            .unflatten(dim=0, sizes=(output_batch_size // 2, 2))
        )
        pullback_metric = torch.cat(
            [pullback_metric, pullback_metric[:, -1:]], dim=1
        ).flatten(0, 1)[:batch_size]
        # pullback_metric.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, embedding_size, embedding_size)

        initial_k = min([embedding_size // 2, g.shape[-1]])
        initial_k = max(
            initial_k, min_num_eigenvectors
        )  # Ensure at least min_num_eigenvectors

        if approximated_eigendecomposition:
            # Now compute final eigendecomposition with determined number of eigenpairs
            eigenvalues, eigenvectors = eigsh(
                pullback_metric,
                k=initial_k,
                which="LM",
                tol=EPS,
                maxiter=initial_k * 100,
                return_eigenvectors=True,
            )

        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric)

        # eigenvalues.shape = (batch_size, max_len, max_len, R)
        # eigenvectors.shape = (batch_size, max_len, max_len, embedding_size, R)
        eigenvalues = eigenvalues.flip(dims=(-1,)).to(dtype=torch.get_default_dtype())
        eigenvectors = eigenvectors.flip(dims=(-1, -2)).to(
            dtype=torch.get_default_dtype()
        )

        if return_trace:
            num_eigenvectors = (
                torch.where(eigenvalues > EPS, 1, 0)
                .sum(dim=-1, keepdim=True)
                .to(eigenvalues.device, dtype=torch.int32)
            )
            num_eigenvectors = torch.clamp(
                num_eigenvectors, min=min_num_eigenvectors, max=max_num_eigenvectors
            )
            # num_eigenvectors.shape = (batch_size, max_len, 1)
            eigenvalues_index = (
                torch.arange(
                    eigenvalues.shape[-1], device=eigenvalues.device, dtype=torch.int32
                )
                .reshape(1, 1, 1, -1)
                .repeat(*eigenvalues.shape[:-1], 1)
            )
            eigenvalues = torch.where(
                eigenvalues_index <= num_eigenvectors,
                eigenvalues,
                torch.zeros_like(eigenvalues),
            )

            traces = eigenvalues.sum(dim=-1)
            radii = eigenvalues[..., 0]
            # `estimate_pdet` keeps a trailing singleton dim (keepdim=True), which
            # would make `special_pdets` 4-D (..., 1) while `traces`/`radii` are 3-D.
            # That extra dim pushes pdets down a different branch of
            # `pad_traces_and_eigenvectors` (cat along dim 0 instead of stacking),
            # mangling its first dimension. Squeeze it so all three line up.
            special_pdets = estimate_pdet(
                eigenvalues, det_regularization=det_regularization, min_rank=min_rank
            ).squeeze(-1)

            eigenvector_index = (
                torch.arange(
                    eigenvectors.shape[-1],
                    device=eigenvectors.device,
                    dtype=torch.int32,
                )
                .reshape(1, 1, 1, 1, -1)
                .repeat(*eigenvectors.shape[:-1], 1)
            )
            eigenvectors = torch.where(
                eigenvector_index <= num_eigenvectors.unsqueeze(-1),
                eigenvectors,
                torch.zeros_like(eigenvectors),
            ).transpose(-1, -2)
            max_index = num_eigenvectors.max().item()
            eigenvector_index = eigenvector_index[..., :max_index, :]
            eigenvectors = torch.gather(
                eigenvectors, dim=-2, index=eigenvector_index
            ).reshape(*eigenvectors.shape[:-2], -1, eigenvectors.shape[-1])

            # shapes:
            # special_pdets.shape = (batch_size, max_len, max_pred_ids)
            # traces.shape = (batch_size, max_len, max_pred_ids)
            # radii.shape = (batch_size, max_len, max_pred_ids)
            # eigenvectors.shape = (batch_size, max_len, max_pred_ids, num_eigenvectors, embedding_size)
            # predictions.shape = (batch_size, max_len, max_pred_ids, selected_vocab_size)
            return (
                special_pdets,
                traces,
                radii,
                eigenvectors,
                (predictions if return_predictions else None),
            )

        if degrowth and same_equivalence_class:
            jac_eigen_dot_prod = torch.bmm(jac, torch.flatten(eigenvectors, end_dim=1))
            # jac_eigen_dot_prod.shape = (batch_size*max_len, output_size, embedding_size)
            jac_eigen_dot_prod = torch.gather(
                jac_eigen_dot_prod,
                dim=1,
                index=predictions.repeat_interleave(max_len)
                .unsqueeze(-1)
                .repeat(1, 1, eigenvectors.shape[-1]),
            )
            jac_eigen_dot_prod = torch.stack(
                torch.chunk(jac_eigen_dot_prod.transpose(0, 1), batch_size)
            )
            jac_eigen_dot_prod = torch.squeeze(
                jac_eigen_dot_prod, dim=list(range(1, jac_eigen_dot_prod.dim()))
            ).reshape(eigenvalues.shape)
            # jac_eigen_dot_prod.shape = (batch_size, max_len, embedding_size)
            eigenvalues = torch.where(
                jac_eigen_dot_prod < 0, eigenvalues, torch.zeros_like(eigenvalues)
            )

    return eigenvalues, eigenvectors, (predictions if return_predictions else None)
