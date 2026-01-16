import torch
from typing import List
from torch.func import jacrev
from transformers import AutoModel
from inspect import signature

EPS = 2*torch.finfo(torch.float32).eps

class OutputOnlyModel(torch.nn.Module):

    def __init__(self, model: AutoModel, encoder_name: str = "encoder"):
        super().__init__()
        self.model = model
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
            
            if "attention_mask" in self.module_signatures["encoder"].parameters:
                X = self.encoder(X, attention_mask = attention_mask)
            elif "attn_mask" in self.module_signatures["encoder"].parameters:
                X = self.encoder(X, attn_mask = attention_mask)
            elif "mask" in self.module_signatures["encoder"].parameters:
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
    min_num_eigenvectors: int = 8,
    max_num_eigenvectors: int = None,
    approximated_eigendecomposition: bool = False,
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
    jac = torch.nn.utils.rnn.pad_sequence([T for i,L in enumerate(eq_class_emb_ids) for T in jac[i,:,L,:]], batch_first = True, padding_value = 0).reshape(batch_size, max_pred_ids, max_len, *jac.shape[-2:])
    # jac.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, selected_vocab_size, embedding_size)

    if max_num_eigenvectors is None:
        max_num_eigenvectors = jac.shape[-1] - 1

    with torch.no_grad():
        jac = torch.transpose(jac, 1, 2)
        jac = torch.flatten(jac, end_dim = -3)
        # jac.shape = (batch_size*max_len*max_len_pred_ids, selected_vocab_size, embedding_size)
        jac_t = torch.transpose(jac, -2, -1)
        # jac_t.shape = (batch_size*max_len, max_len_pred_ids, embedding_size, selected_vocab_size)

        # g.shape = (batch_size, max_len, selected_vocab_size, selected_vocab_size)
        tmp = torch.bmm(jac_t, torch.flatten(g, end_dim=1)) if g.dim() == 4 else torch.bmm(jac_t, g)
        # tmp.shape = (batch_size*max_len*max_len_pred_ids, embedding_size, selected_vocab_size)
        pullback_metric = torch.bmm(tmp, jac)
        # pullback_metric.shape = (batch_size*max_len*max_len_pred_ids, embedding_size, embedding_size)

        pullback_metric = pullback_metric.reshape(batch_size, max_pred_ids, max_len, *pullback_metric.shape[-2:]).to(dtype = torch.float32)
        # pullback_metric.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, embedding_size, embedding_size)

        # Strategy 1: Use partial eigendecomposition to estimate rank instead of full matrix_rank
        # This is faster because we reuse the eigendecomposition work
        embedding_size = pullback_metric.shape[-1]
        initial_k = min(min_num_eigenvectors * 2, embedding_size // 2)
        initial_k = max(initial_k, min_num_eigenvectors)  # Ensure at least min_num_eigenvectors

        if approximated_eigendecomposition:
            # Do initial eigendecomposition with looser tolerance to estimate rank
            try:
                eigenvalues_init, eigenvectors_init = torch.lobpcg(
                    pullback_metric,
                    k=initial_k,
                    method="ortho",
                    tol=EPS * 100,  # Looser tolerance for rank estimation
                    niter=min_num_eigenvectors * 10
                )
                
                # Estimate rank by counting eigenvalues above threshold
                # Use relative threshold based on maximum eigenvalue magnitude
                eigenvalue_magnitude = torch.abs(eigenvalues_init)
                max_eigenval = eigenvalue_magnitude.max(dim=-1, keepdim=True)[0]
                # Threshold: eigenvalues must be at least 1e-4 of the maximum
                eigenvalue_threshold = max_eigenval * 1e-4
                effective_rank = (eigenvalue_magnitude > eigenvalue_threshold).sum(dim=-1)
                
                # Determine num_eigenpairs from estimated rank
                num_eigenpairs = min(
                    max(min_num_eigenvectors, int(effective_rank.median().item())),
                    embedding_size // 3 - 1
                )
            except RuntimeError:
                # Fallback: if initial eigendecomposition fails, use conservative estimate
                num_eigenpairs = min(
                    max(min_num_eigenvectors, embedding_size // 4),  # Conservative estimate
                    embedding_size // 3 - 1
                )
                eigenvectors_init = torch.zeros_like(pullback_metric)[...,:num_eigenpairs]

            assert num_eigenpairs  <= initial_k, f"num_eigenpairs ({num_eigenpairs}) must be less than or equal to initial_k ({initial_k})"

            # Now compute final eigendecomposition with determined number of eigenpairs
            eigenvalues, eigenvectors = torch.lobpcg(pullback_metric, k = num_eigenpairs, X = eigenvectors_init, method = "ortho", tol = EPS, niter = min_num_eigenvectors*20)
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric)
            eigenvalues = eigenvalues
            eigenvectors = eigenvectors

        # eigenvalues.shape = (batch_size, max_len, max_len, R)
        # eigenvectors.shape = (batch_size, max_len, max_len, embedding_size, R)
        eigenvalues = eigenvalues[...,::-1]
        eigenvectors = eigenvectors[...,::-1]

        if return_trace:
            num_eigenvectors = torch.where(eigenvalues > EPS, 1, 0).sum(dim = -1, keepdim = True).to(eigenvalues.device, dtype = torch.int32)
            num_eigenvectors = torch.clamp(num_eigenvectors, min = min_num_eigenvectors, max = max_num_eigenvectors)
            # num_eigenvectors.shape = (batch_size, max_len, 1)
            eigenvalues_index = torch.arange(eigenvalues.shape[-1], device = eigenvalues.device, dtype = torch.int32).reshape(1,1,1,-1).repeat(*eigenvalues.shape[:-1], 1)
            eigenvalues = torch.where(eigenvalues_index <= num_eigenvectors, eigenvalues, torch.zeros_like(eigenvalues))
            trace = torch.sum(eigenvalues, dim = -1)
            pseudodeterminant = torch.prod(eigenvalues[eigenvalues > EPS], dim = -1)
            spectral_radius = eigenvalues[..., 0]

            special_trace = 0.5*trace + 0.5*pseudodeterminant/spectral_radius

            eigenvector_index = torch.arange(eigenvectors.shape[-1], device = eigenvectors.device, dtype = torch.int32).reshape(1,1,1,1,-1).repeat(*eigenvectors.shape[:-1], 1)
            eigenvectors = torch.where(eigenvector_index <= num_eigenvectors.unsqueeze(-1), eigenvectors, torch.zeros_like(eigenvectors))
            eigenvectors = eigenvectors[...,:num_eigenvectors.max().item()]

            return special_trace, eigenvectors, (predictions if return_predictions else None)

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