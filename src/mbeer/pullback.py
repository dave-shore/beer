import torch
from typing import List
from torch.func import jacrev
from transformers import AutoModel
from inspect import signature

EPS = 2*torch.finfo(torch.float32).eps

class OutputOnlyModel(torch.nn.Module):

    def __init__(self, model: AutoModel):
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor, pred_id: List[int] = None, select: torch.Tensor = None, attention_mask = None, given_predictions: torch.Tensor = None):
        if hasattr(self.model, "base_model"):
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(X.shape[0], X.shape[1])
            
            for n,module in self.model.base_model.named_children():
                if n not in ["embeddings", "embed_tokens"]:
                    module_signature = signature(module.forward)
                    if "attention_mask" in module_signature.parameters:
                        X = module(X, attention_mask = attention_mask)
                    elif "attn_mask" in module_signature.parameters:
                        X = module(X, attn_mask = attention_mask)
                    else:
                        X = module(X)

            encoder_output = X.last_hidden_state if hasattr(X, "last_hidden_state") else X

            if hasattr(self.model, "classifier"):
                y = self.model.classifier(
                    encoder_output[:,pred_id,:] if encoder_output.dim() == 3 else encoder_output
                )
            elif hasattr(self.model, "cls"):
                y = self.model.cls(
                    encoder_output[:,pred_id,:] if encoder_output.dim() == 3 else encoder_output
                )
            else:
                raise AttributeError("Model has neither classifier nor cls head")
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
            return y, torch.max(torch.softmax(y, dim = -1), dim = -1).values

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
    batch_jacobian, predictions = list(zip(*[
        jacrev(model, argnums = 0, has_aux=True)(
            x,
            pred_id[i],
            select[i],
            None if attention_mask is None else attention_mask[i],
            None if given_predictions is None else given_predictions[i]
        )
    for i,x in enumerate(nn_input.split(1))]))

    max_len_pred_ids = max(map(len, pred_id))
    batch_jacobian = torch.stack([
        torch.cat((J, torch.zeros(J.shape[0], max_len_pred_ids - J.shape[1], *J.shape[2:], device = J.device, dtype = J.dtype)), dim = 1)
        if J.shape[1] < max_len_pred_ids else 
        J
    for J in batch_jacobian])
    predictions = torch.nn.utils.rnn.pad_sequence([p.T for p in predictions], batch_first = True, padding_value = 0)
    # batch_jacobian.shape = (batch_size, output_size, (N_patches+1), embedding_size)

    batch_jacobian = torch.squeeze(batch_jacobian, dim = list(range(1,batch_jacobian.dim())))
    predictions = torch.squeeze(predictions, dim = list(range(1,predictions.dim())))
    # batch_jacobian.shape = (batch_size, max_len_pred_ids, selected_vocab_size, input_seq_len, embedding_size)
    # predictions.shape = (batch_size, max_len_pred_ids)

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

    jac, predictions = jacobian(input_simec, model, select=select, pred_id = pred_id, attention_mask = attention_mask, given_predictions = given_predictions)
    # jac.shape = (batch_size, max_len_pred_ids, selected_vocab_size, input_seq_len, embedding_size)
    # predictions.shape = (batch_size,)
    while jac.dim() < 5:
        jac = torch.unsqueeze(jac, 0)
    
    jac = jac.reshape(
        input_simec.shape[0],
        max(map(len, pred_id)),
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
    jac = torch.stack(
        [
            torch.cat((jac[i,:,L,:], torch.zeros(jac.shape[1], max_len - len(L), *jac.shape[3:], device = jac.device, dtype = jac.dtype)), dim = 1)
            if len(L) < max_len else 
            jac[i,:,L,:]
        for i,L in enumerate(eq_class_emb_ids)]
    )
    # jac.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, selected_vocab_size, embedding_size)

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

        pullback_metric = torch.stack(
            torch.chunk(pullback_metric, batch_size)
        )
        pullback_metric = torch.stack(
            torch.chunk(pullback_metric, max(map(len, pred_id)), dim = 1),
            dim = 1
        )
        # pullback_metric.shape = (batch_size, max_len_pred_ids, max_eq_class_emb_ids_length, embedding_size, embedding_size)

        pullback_metric = pullback_metric.to(dtype = torch.float32)

        R = torch.linalg.matrix_rank(pullback_metric, hermitian = True)

        eigenvalues, eigenvectors = torch.lobpcg(pullback_metric, k = max(min_num_eigenvectors, int(R.median().item())), method = "ortho", tol = EPS, niter = min_num_eigenvectors*20)
        # eigenvalues.shape = (batch_size, max_len, R)
        # eigenvectors.shape = (batch_size, max_len, embedding_size, R)

        if return_trace:
            trace = torch.sum(eigenvalues, dim = -1)
            return trace, eigenvectors, (predictions if return_predictions else None)

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