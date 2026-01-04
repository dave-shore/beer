import torch
from typing import List
from torch.func import jacrev
from transformers import AutoModel

EPS = 2*torch.finfo(torch.float32).eps

class OutputOnlyModel(torch.nn.Module):

    def __init__(self, model: AutoModel):
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor, pred_id: List[int] = None, select: torch.Tensor = None, attention_mask = None):
        if hasattr(self.model, "base_model"):
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(X.shape[0], X.shape[1], 1)
            encoder_output = self.model.base_model.encoder(X, attention_mask = attention_mask)['last_hidden_state']

            if hasattr(self.model, "pooler"):
                encoder_output = self.model.bert.pooler(encoder_output)

            if hasattr(self.model, "classifier"):
                y = self.model.classifier(
                    encoder_output
                )

            elif hasattr(self.model, "cls"):
                y = self.model.cls(
                    encoder_output[:,pred_id,:]
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

        return y, torch.argmax(y, dim = -1)

def jacobian(nn_input: torch.Tensor, model: torch.nn.Module = None, select: torch.Tensor = None, pred_id: List[int] = None, attention_mask = None):
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
            None if attention_mask is None else attention_mask[i]
        )
    for i,x in enumerate(nn_input.split(1))]))
    batch_jacobian = torch.stack(batch_jacobian)
    predictions = torch.stack(predictions)
    # batch_jacobian.shape = (batch_size, output_size, (N_patches+1), embedding_size)

    batch_jacobian = torch.squeeze(batch_jacobian, dim = list(range(1,batch_jacobian.dim())))
    predictions = torch.squeeze(predictions, dim = list(range(1,predictions.dim())))

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

    jac, predictions = jacobian(input_simec, model, select=select, pred_id = pred_id, attention_mask = attention_mask)
    # jac.shape = (batch_size, output_size, N_patches+1, embedding_size)
    # predictions.shape = (batch_size,)
    while jac.dim() < 4:
        jac = torch.unsqueeze(jac, 0)

    # Select ids and pad if necessary
    max_len = max(map(len, eq_class_emb_ids))
    batch_size = jac.shape[0]
    jac = torch.stack(
        [
            torch.nn.functional.pad(jac[i,:,L,:], (0,0,0,max_len-len(L)))
        for i,L in enumerate(eq_class_emb_ids)]
    )
    # jac.shape = (batch_size, output_size, max_len, embedding_size)

    with torch.no_grad():
        jac = torch.transpose(jac, 1, 2)
        jac = torch.flatten(jac, end_dim = 1)
        # jac.shape = (batch_size*max_len, output_size, embedding_size)
        jac_t = torch.transpose(jac, -2, -1)
        # jac_t.shape = (batch_size*max_len, embedding_size, output_size)

        # g.shape = (batch_size, max_len, output_size, output_size)
        tmp = torch.bmm(jac_t, torch.flatten(g, end_dim=1))
        # tmp.shape = (batch_size*max_len, embedding_size, output_size)
        pullback_metric = torch.bmm(tmp, jac)
        # pullback_metric.shape = (batch_size*max_len, embedding_size, embedding_size)

        pullback_metric = torch.stack(
            torch.chunk(pullback_metric, batch_size)
        )
        # pullback_metric.shape = (batch_size, max_len, embedding_size, embedding_size)

        R = torch.linalg.matrix_rank(pullback_metric, tol = EPS, hermitian = True)
        m = pullback_metric.shape[-1]

        eigenvalues, eigenvectors = torch.linalg.lobpcg(pullback_metric, k = m - R, method = "ortho")
        # eigenvalues.shape = (batch_size, max_len, m - R)
        # eigenvectors.shape = (batch_size, max_len, embedding_size, m - R)

        if return_trace:
            trace = torch.sum(eigenvalues, dim = -1)
            return trace, eigenvectors

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

    return eigenvalues, eigenvectors