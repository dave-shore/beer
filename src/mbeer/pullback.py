import torch
from typing import List
from torch.func import jacrev
from transformers import AutoModel
from torchkde import KernelDensity
from cupyx.scipy.sparse.linalg import eigsh
import math
from inspect import signature
import warnings

EPS = 2*torch.finfo(torch.float32).eps

class OutputOnlyModel(torch.nn.Module):

    def __init__(self, model: AutoModel, encoder_name: str = "encoder"):
        super().__init__()
        self.model = model
        self.encoder_name = encoder_name
        self.device = model.device if hasattr(model, "device") else next(model.parameters()).device
        self.dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype
        self.encoder = model.base_model.get_submodule(encoder_name)

        self.module_signatures = {n:signature(module.forward) for n,module in self.model.base_model.named_children()}
        self.lm_head = torch.nn.Sequential(*[getattr(self.model, n, torch.nn.Identity()) for n in [n for n,module in self.model.named_children() if n in ["classifier", "cls", "lm_head", "generator_lm_head", "generator_predictions"]]])

    def forward(self, X: torch.Tensor, pred_id: List[int] = None, select: torch.Tensor = None, attention_mask = None, given_predictions: torch.Tensor = None):
        if hasattr(self.model, "base_model"):

            if X.dim() < 3:
                X = X.unsqueeze(0)

            batch_size, seq_len, embedding_size = X.shape
            if attention_mask is not None:
                if attention_mask.dim() == 1 or (attention_mask.dim() == 2 and attention_mask.shape[1] != seq_len):
                    attention_mask = attention_mask.reshape(1,-1).expand(batch_size, -1)
                elif attention_mask.shape[0] < batch_size:
                    attention_mask = attention_mask[[0]*batch_size].reshape(batch_size, seq_len)
                else:
                    attention_mask = attention_mask[:batch_size].reshape(batch_size, seq_len)
            
            if "attention_mask" in self.module_signatures[self.encoder_name].parameters:
                X = self.encoder(X, attention_mask = attention_mask)
            elif "attn_mask" in self.module_signatures[self.encoder_name].parameters and "distil" not in self.model.config._name_or_path:
                X = self.encoder(X, attn_mask = attention_mask)
            elif "mask" in self.module_signatures[self.encoder_name].parameters:
                X = self.encoder(X, mask = attention_mask)
            else:
                X = self.encoder(X)

            encoder_output = X.last_hidden_state if hasattr(X, "last_hidden_state") else X

            encoder_output = encoder_output[:,pred_id,:] if encoder_output.dim() == 3 else encoder_output
            y = self.lm_head(encoder_output)

        else:
            encoder_output, _ = self.model.encoder(X)
            y = self.model.classifier(encoder_output[:, 0])

        if select is not None:
            select = select.unsqueeze(0)
            y = torch.gather(
                y,
                dim = -1,
                index = select[:,pred_id,:]
            )

        if given_predictions is not None:
            pred_proba = torch.gather(
                torch.softmax(y, dim = -1),
                dim = -1,
                index = given_predictions.unsqueeze(-1) if given_predictions.dim() < y.dim() else given_predictions
            )
            return y, pred_proba

        else:
            return y, torch.softmax(y, dim = -1)

def jacobian(nn_input: torch.Tensor, model: torch.nn.Module = None, select: torch.Tensor = None, pred_id: List[int] = None, attention_mask = None, given_predictions: torch.Tensor = None):
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
        pred_id = [0]*nn_input.shape[0]

    if select is None:
        select = [None]*nn_input.shape[0]

    # nn_input.shape = (batch_size, N_patches+1, embedding_size)
    batch_jacobian_and_predictions = [
        jacrev(model, argnums = 0, has_aux=True)(
            x,
            pred_id[i],
            select[i],
            None if attention_mask is None else attention_mask[i],
            None if given_predictions is None else given_predictions[i]
        )
    for i,x in enumerate(nn_input.split(1))]
    nn_input.grad = None

    batch_jacobian = torch.nn.utils.rnn.pad_sequence([J.squeeze(0) for J, _ in batch_jacobian_and_predictions], batch_first = True, padding_value = 0)
    predictions = torch.nn.utils.rnn.pad_sequence([p.squeeze(0) for _, p in batch_jacobian_and_predictions], batch_first = True, padding_value = 0)
    # batch_jacobian.shape = (batch_size, output_size, (N_patches+1), embedding_size)

    batch_jacobian = torch.squeeze(batch_jacobian, dim = list(range(1,batch_jacobian.dim())))
    predictions = torch.squeeze(predictions, dim = list(range(1,predictions.dim())))
    # batch_jacobian.shape = (batch_size, max_len_pred_ids, selected_vocab_size, input_seq_len, embedding_size)
    # predictions.shape = (batch_size, max_len_pred_ids, selected_vocab_size)

    return batch_jacobian, predictions


def kde_filter_rank_estimate(eigenvalues: torch.Tensor, granularity: float = 1e-4, skip_first_n: int = 5, cutoff: float = 10) -> float:

    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    y = eigenvalues
    y = y[y < cutoff] if cutoff else y
    kde.fit(y.reshape(-1,1))

    x = torch.linspace(0, 1, int(granularity**(-1)))[skip_first_n:-skip_first_n]
    prob = kde.score_samples(x.reshape(-1,1).to(eigenvalues.device, dtype =eigenvalues.dtype)).exp()

    diffs = torch.diff(prob, n = 1, prepend = torch.ones_like(prob[[0]]))
    second_diffs = torch.diff(prob, n = 2, prepend = torch.ones_like(prob[:2]))

    threshold = x[torch.logical_and(diffs.cpu() > -EPS, second_diffs.cpu() > -EPS)][0].item()

    r = torch.nn.functional.threshold(eigenvalues, threshold, 0).sum(dim = -1)

    if torch.isnan(r).any():
        warnings.warn("NaN values in r")

    return r


def chebyshev_filter_rank_estimate(matrix: torch.Tensor, eigenvalues: torch.Tensor, number_of_sample_vectors: int, degree: int) -> float:

    random_sample_vectors = torch.randn(number_of_sample_vectors, eigenvalues.shape[-1], device = eigenvalues.device, dtype = eigenvalues.dtype)
    random_sample_vectors = random_sample_vectors / torch.linalg.norm(random_sample_vectors, dim = -1, keepdim = True)    

    dos = torch.cos(eigenvalues)**(-1)
    dos_diff = dos.diff(dim = -1)
    a = torch.min(eigenvalues[dos_diff > -EPS], dim = -1)[0]
    b = torch.max(eigenvalues, dim = -1)[0]
    gamma_coefficients = [eigenvalues.shape[-1] * (torch.cos(a)**(-1) + torch.cos(b)**(-1)) // math.pi] + [
        eigenvalues.shape[-1] *(2/math.pi) * (torch.sin(k*torch.cos(a)**(-1)) + torch.sin(k*torch.cos(b)**(-1))) / k
    for k in range(1, degree+1)]

    quadratic_forms = [
        torch.sum(
            (torch.special.chebyshev_polynomial_t(matrix, k) @ random_sample_vectors.T) * random_sample_vectors.T,
            dim = -2
        ).mean(dim = -1)
    for k in range(degree+1)]

    r = torch.sum(torch.stack(quadratic_forms) * torch.stack(gamma_coefficients), dim = 0)
    
    if torch.isnan(r).any():
        warnings.warn("NaN values in r")

    return r

@torch.no_grad()
def estimate_pdet(eigenvalues: torch.Tensor, det_regularization: float = 1e-3) -> torch.Tensor:

    tr_rad_ratio = eigenvalues.sum(dim = -1) / (eigenvalues[...,0] + EPS)
    tr_sq_rad_ratio = eigenvalues.pow(2).sum(dim = -1) / (eigenvalues.pow(2)[...,0] + EPS)
    d = eigenvalues.shape[-1]

    # Mid-point coefficient estimate for the pseudo-determinant
    normalized_eigenvalues = eigenvalues / (eigenvalues[...,[0]] + EPS)
    rk_est = torch.sum(normalized_eigenvalues, dim = -1)
    rk_est = torch.clamp(rk_est, min = 1, max = d)
    
    log_lb = torch.sum(torch.log(eigenvalues + det_regularization), dim = -1) - (d - rk_est) * math.log(det_regularization) 

    log_ub = log_lb + det_regularization*tr_rad_ratio + (1-det_regularization)*tr_sq_rad_ratio / 2

    general_ub = (eigenvalues.sum(dim = -1) / rk_est)**rk_est

    lb = torch.exp(log_lb)
    lb = torch.clamp(lb, min = EPS*torch.ones_like(lb), max = torch.maximum(EPS*torch.ones_like(lb), general_ub))
    ub = torch.exp(log_ub)
    ub = torch.clamp(ub, min = lb, max = torch.maximum(lb, general_ub))

    special_pdets = torch.where(
        torch.isfinite(ub), 
        (ub + lb) / 2, 
        lb
    )

    return special_pdets
    
    

def pullback(
    input_simec: torch.Tensor,
    g: torch.Tensor,
    model: torch.nn.Module = None,
    eq_class_emb_ids: List[List[int]] = None,
    select: torch.Tensor = None,
    pred_id: List[int] = None,
    degrowth: bool = True,
    same_equivalence_class: bool = True,
    attention_mask = None,
    return_trace: bool = False,
    return_predictions: bool = False,
    given_predictions: torch.Tensor = None,
    min_num_eigenvectors: int = 32,
    max_num_eigenvectors: int = None,
    approximated_eigendecomposition: bool = False,
    det_regularization: float = 1e-3
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

    input_simec = input_simec.to(dtype = torch.float64)
    g = g.to(dtype = torch.float64)
    if given_predictions is not None:
        given_predictions = given_predictions.to(dtype = torch.float64)
    if model is not None:
        model = model.to(dtype = torch.float64)

    if pred_id is None:
        pred_id = [0]*input_simec.shape[0]

    pred_id = [list(set(p)) for p in pred_id]

    jac, predictions = jacobian(input_simec, model, select=select, pred_id = pred_id, attention_mask = attention_mask, given_predictions = given_predictions)
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
        input_simec.shape[-1]
    )

    if select is not None:
        jac = torch.transpose(jac, 2, 3)
        # jac.shape = (batch_size, max_len_pred_ids, input_seq_len, selected_vocab_size, embedding_size)

    # Select ids and pad if necessary
    max_len = max(map(len, eq_class_emb_ids))
    batch_size = jac.shape[0]
    max_idx = max(max(L) for L in eq_class_emb_ids)
    assert max_idx < jac.shape[2], f"eq_class_emb_ids index {max_idx} >= jac dim 2 ({jac.shape[2]})"

    jac = torch.nn.utils.rnn.pad_sequence([T for i,L in enumerate(eq_class_emb_ids) for T in jac[i,:,L,:]], batch_first = True, padding_value = 0).reshape(batch_size, max_pred_ids, max_len, *jac.shape[-2:])
    # jac.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, selected_vocab_size, embedding_size)

    if max_num_eigenvectors is None:
        max_num_eigenvectors = jac.shape[-1] - 1

    assert input_simec.shape[0] % 3 == 0, f"Batch size {input_simec.shape[0]} must be divisible by 3"

    #########################################################
    jac = torch.stack([torch.cat([chunk, chunk[-1:]], dim=0) for chunk in torch.split(jac, 3, dim = 0)]).transpose(1, 3).flatten(start_dim = 3, end_dim = 4)
    # J_grouped.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, 4*selected_vocab_size, embedding_size)
    jac_I = torch.cat(torch.chunk(jac, 4, dim = -2)[::2], dim = -2)
    jac_J = torch.cat(torch.chunk(jac, 4, dim = -2)[1::2], dim = -2)
    # jac_I.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, 4*selected_vocab_size, embedding_size)
    # jac_J.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, 4*selected_vocab_size, embedding_size)

    jac = torch.stack([jac_I, jac_J], dim = 1).flatten(end_dim = 1)
    # jac.shape = (batch_size*2, max_len_pred_ids, max_eq_class_emb_ids_length, 2*selected_vocab_size, embedding_size)
    g = g.repeat_interleave(2, dim = -1).repeat_interleave(2, dim = -2)

    with torch.no_grad():
        jac = torch.transpose(jac, 1, 2)
        jac = torch.flatten(jac, end_dim = -3)
        # jac.shape = (batch_size*max_len*max_len_pred_ids, selected_vocab_size, embedding_size)

        # g.shape = (batch_size, max_len, selected_vocab_size, selected_vocab_size)
        g_flat = torch.flatten(g, end_dim = 1) if g.dim() == 4 else g
        g_flat = g_flat[:jac.shape[0]] if g.shape[0] > jac.shape[0] else g_flat.tile(jac.shape[0] // g.shape[0] + 1, 1, 1)[:jac.shape[0]]
        if jac.shape[-2] < g_flat.shape[-1]:
            jac = torch.nn.functional.pad(jac, (0,0,0,g_flat.shape[-1]-jac.shape[-2]))
        elif jac.shape[-1] > g_flat.shape[-1]:
            g_flat = torch.nn.functional.pad(g_flat, (0,jac.shape[-2]-g_flat.shape[-1],0,0))
        
        if jac.shape[0] < g_flat.shape[0]:
            jac = torch.cat([jac, torch.zeros(g_flat.shape[0]-jac.shape[0], *jac.shape[1:], device = jac.device, dtype = jac.dtype)])

        jac_t = torch.transpose(jac, -2, -1)
        # jac_t.shape = (batch_size*max_len, max_len_pred_ids, embedding_size, selected_vocab_size)

        tmp = torch.bmm(jac_t, g_flat)
        # tmp.shape = (batch_size*max_len*max_len_pred_ids, embedding_size, selected_vocab_size)
        pullback_metric = torch.bmm(tmp, jac)
        # pullback_metric.shape = (batch_size*max_len*max_len_pred_ids, embedding_size, embedding_size)

        output_batch_size = batch_size // 3 * 2
        output_batch_size = max(1, output_batch_size)

        try:
            pullback_metric = pullback_metric.reshape(output_batch_size, max_pred_ids, max_len, *pullback_metric.shape[-2:])
        except RuntimeError:
            pullback_metric = pullback_metric.reshape(2*output_batch_size, max_pred_ids, max_len, *pullback_metric.shape[-2:])[::2]
        # pullback_metric.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, embedding_size, embedding_size)

        embedding_size = pullback_metric.shape[-1]
        initial_k = min([embedding_size // 2, g.shape[-1]])
        initial_k = max(initial_k, min_num_eigenvectors)  # Ensure at least min_num_eigenvectors

        if approximated_eigendecomposition:
            # Now compute final eigendecomposition with determined number of eigenpairs
            eigenvalues, eigenvectors = eigsh(pullback_metric, k = initial_k, which = "LM", tol = EPS, maxiter = initial_k*100, return_eigenvectors = True)

        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric)

        # eigenvalues.shape = (batch_size, max_len, max_len, R)
        # eigenvectors.shape = (batch_size, max_len, max_len, embedding_size, R)
        eigenvalues = eigenvalues.flip(dims = (-1,))
        eigenvectors = eigenvectors.flip(dims = (-1,-2))

        if return_trace:
            num_eigenvectors = torch.where(eigenvalues > EPS, 1, 0).sum(dim = -1, keepdim = True).to(eigenvalues.device, dtype = torch.int32)
            num_eigenvectors = torch.clamp(num_eigenvectors, min = min_num_eigenvectors, max = max_num_eigenvectors)
            # num_eigenvectors.shape = (batch_size, max_len, 1)
            eigenvalues_index = torch.arange(eigenvalues.shape[-1], device = eigenvalues.device, dtype = torch.int32).reshape(1,1,1,-1).repeat(*eigenvalues.shape[:-1], 1)
            eigenvalues = torch.where(eigenvalues_index <= num_eigenvectors, eigenvalues, torch.zeros_like(eigenvalues))

            traces = eigenvalues.sum(dim = -1)
            radii = eigenvalues[...,0]
            special_pdets = estimate_pdet(eigenvalues, det_regularization = det_regularization)

            eigenvector_index = torch.arange(eigenvectors.shape[-1], device = eigenvectors.device, dtype = torch.int32).reshape(1,1,1,1,-1).repeat(*eigenvectors.shape[:-1], 1)
            eigenvectors = torch.where(eigenvector_index <= num_eigenvectors.unsqueeze(-1), eigenvectors, torch.zeros_like(eigenvectors))
            eigenvectors = eigenvectors[...,:num_eigenvectors.max().item()]

            return special_pdets, traces, radii, eigenvectors, (predictions if return_predictions else None)

        if degrowth and same_equivalence_class:
            jac_eigen_dot_prod = torch.bmm(jac, torch.flatten(eigenvectors, end_dim=1))
            # jac_eigen_dot_prod.shape = (batch_size*max_len, output_size, embedding_size)
            jac_eigen_dot_prod = torch.gather(jac_eigen_dot_prod, dim = 1, index = predictions.repeat_interleave(max_len).unsqueeze(-1).repeat(1,1,eigenvectors.shape[-1]))
            jac_eigen_dot_prod = torch.stack(
                torch.chunk(jac_eigen_dot_prod.transpose(0,1), batch_size)
            )
            jac_eigen_dot_prod = torch.squeeze(jac_eigen_dot_prod, dim = list(range(1,jac_eigen_dot_prod.dim()))).reshape(eigenvalues.shape)
            # jac_eigen_dot_prod.shape = (batch_size, max_len, embedding_size)
            eigenvalues = torch.where(jac_eigen_dot_prod < 0, eigenvalues, torch.zeros_like(eigenvalues))

    return eigenvalues, eigenvectors, (predictions if return_predictions else None) 