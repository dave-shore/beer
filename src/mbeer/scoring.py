from types import NoneType
from typing import Any, List, Dict
import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations, chain
import pandas as pd
import json
from math import sqrt, log
from scipy.stats import chi2
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm, trange
from copy import deepcopy
import re
import gc
import os

hf_api = HfApi()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EPS = 2*torch.finfo(torch.float32).eps

# Handle both relative and absolute imports
try:
    from .pullback import OutputOnlyModel, pullback
    from .utils import timing_decorator, batch_generator
except ImportError:
    # If relative import fails, try absolute import
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from mbeer.pullback import OutputOnlyModel, pullback
    from mbeer.utils import timing_decorator, batch_generator


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
            L = [T.reshape(tuple(s if s in force_shape else -1 for s in T.shape)) for T in L]
        except:
            raise ValueError("All tensors must have the same number of dimensions or be reshapeable to the same number of dimensions")

    if dim < 3:
        max_last_dim = max(T.shape[-1] for T in L)
        return torch.nn.utils.rnn.pad_sequence([T.unsqueeze(-1) if T.dim() == 1 else torch.nn.functional.pad(T, (0,max_last_dim-T.shape[1])) for T in L], batch_first = True, padding_value = 0)

    padding_shape = tuple(max(along_dim) for along_dim in zip(*shapes))

    for i,T in enumerate(L):
        if i % 3:
            # For pairs (j,i), with i < j, we pad the left side of the output dim (dim 1) and the right side of the input dim (dim 2)
            left_padding = tuple(chain.from_iterable([[padding_shape[j]-T.shape[j],0] if j % 2 else [0,padding_shape[j]-T.shape[j]] for j in range(len(T.shape))][::-1]))
            T = torch.nn.functional.pad(T, left_padding).to(dtype = torch.float32)
        else:
            # For pairs (j,i), with i > j, we pad the right side of the output dim (dim 1) and the left side of the input dim (dim 2)
            right_padding = tuple(chain.from_iterable([[0,padding_shape[j]-T.shape[j]] if j % 2 else [padding_shape[j]-T.shape[j],0] for j in range(len(T.shape))][::-1]))
            T = torch.nn.functional.pad(T, right_padding).to(dtype = torch.float32)

        # For tensors that combine two entities it doesn't matter, we treat them as left-padded
        L[i] = T

    return torch.cat(L)
    

def train_lm_head(model_name: str, device: torch.device, mask_id: int, n_epochs: int = 10, inputs: torch.Tensor = None, attn_mask: torch.Tensor = None):

    parent_model_info = hf_api.model_info(model_name)
    parent_model_name = parent_model_info.card_data.get("base_model", None)
    ignore_modules = ["dropout", "activation", "act_fn"]

    masked_inputs = torch.where(torch.rand_like(inputs.to(dtype = torch.float32)) < 0.15, mask_id, inputs)

    if parent_model_name is not None:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(parent_model_name).to(device, dtype = torch.float32))

    else:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(model_name).to(device, dtype = torch.float32))
        are_weights_initialized ={k:all(torch.all(p.diff() < 1e-6) for p in v.parameters()) for k,v in output_only_model.model.named_modules() if hasattr(v, "parameters") and all(ign not in k for ign in ignore_modules)}
        uninitialized_modules = [k for k,v in are_weights_initialized.items() if v]
        uninitialized_parameters = [p for v in uninitialized_modules for p in output_only_model.model.get_submodule(v).parameters()]

        if len(uninitialized_modules) > 0:

            if os.path.exists(os.path.join("models", model_name.split("/")[-1])):
                for k in uninitialized_modules:
                    output_only_model.model.get_submodule(k).load_state_dict(torch.load(os.path.join("models", model_name.split("/")[-1], f"{k}.pt")))

                return output_only_model

            os.makedirs(os.path.join("models", model_name.split("/")[-1]), exist_ok = True)

            with torch.enable_grad():
                for module in uninitialized_modules:
                    output_only_model.model.get_submodule(module).train()

                # One step of gradient descent
                loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")
                optimizer = torch.optim.LBFGS(uninitialized_parameters, lr = 1e-3, history_size = n_epochs, max_iter = n_epochs)

                dataset = TensorDataset(masked_inputs, inputs, attn_mask)
                dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, pin_memory = True, num_workers = 2)

                def closure(selected_inputs: torch.Tensor, selected_targets: torch.Tensor, selected_attn_mask: torch.Tensor):
                    optimizer.zero_grad()
                    y = output_only_model.model(selected_inputs.to(device), attention_mask = selected_attn_mask.to(device)).logits
                    loss = loss_fn(
                        torch.nn.functional.one_hot(selected_targets.to(device), num_classes = output_only_model.model.config.vocab_size).to(y.device, dtype = y.dtype),
                        torch.nn.functional.softmax(y, dim = -1),
                    )
                    loss.backward()
                    return loss

                for epoch in trange(n_epochs, desc = "Training LM head"):
                    for minibatch in dataloader:
                        
                        selected_inputs, selected_targets, selected_attn_mask = minibatch

                        optimizer.step(lambda: closure(selected_inputs, selected_targets, selected_attn_mask))

            optimizer.zero_grad(set_to_none = True)
            del optimizer, loss_fn

            for module in uninitialized_modules:
                output_only_model.model.get_submodule(module).eval()
                torch.save(output_only_model.model.get_submodule(module).state_dict(), os.path.join("models", model_name.split("/")[-1], f"{module}.pt"))

    return output_only_model


@torch.no_grad()
@timing_decorator
def forward_backward_pass(batch: List[str], model: AutoModelForTokenClassification, tokenizer: AutoTokenizer, mlm_model: AutoModelForMaskedLM, min_num_eigenvectors: int = 8, max_num_eigenvectors: int = None, max_tokens_per_entity: int = 3, max_retained_vocab: int = 30, n_epochs: int = 10):

    max_len = model.config.max_position_embeddings - 2 if hasattr(model.config, "max_position_embeddings") else model.config.n_ctx - 2
    pred_to_mask = {k:int(v.startswith(('B','U'))) + 2*int(v.startswith(('I','L')) and not v.endswith('0')) for k,v in model.config.id2label.items()} if hasattr(model.config, "id2label") else {}
    device = model.device
    mlm_model = mlm_model.model.to("cpu") if isinstance(mlm_model, OutputOnlyModel) else mlm_model.to("cpu")

    encoded_inputs = tokenizer(batch, padding = "max_length", add_special_tokens = False, max_length = max_len, truncation=True, return_tensors = "pt")
    attn_mask = encoded_inputs.attention_mask
    row_lens = attn_mask.sum(dim = 1)
    num_labels = len(pred_to_mask) if pred_to_mask else 2
    # Threshold from power of Chi-Squared test for uniformity of distribution
    probit_threshold = 1 / num_labels + sqrt(chi2.sf(0.95, num_labels - 1) / (2*num_labels))
    probit_threshold = min(probit_threshold, 0.5)

    logit_threshold = torch.nn.Sequential(torch.nn.Softmax(dim = -1), torch.nn.Threshold(probit_threshold, 0))

    raw_predictions = model(**encoded_inputs.to(device)).logits
    predictions = logit_threshold(raw_predictions * attn_mask.unsqueeze(-1).to(raw_predictions.device)).cpu()
    if predictions.shape[-1] > 1:
        predictions = predictions.argmax(dim = -1)
    else:
        predictions = predictions.round().to(dtype = torch.int8).squeeze(-1)
    # predictions.shape = (batch_size, max_len)

    # Get the entity spans as (start, end) tuples
    pred_mask = predictions.apply_(pred_to_mask.get) if len(pred_to_mask) > 2 else torch.where(predictions > 0, 1, 0)
    ent_spans = [
        torch.logical_and(
            torch.diff(seq, prepend = torch.tensor([0])) != 0,
            torch.logical_or(
                torch.roll(seq, shifts = -1) != 2,
                torch.roll(seq, shifts = +1) == 0
            )
        ).to(dtype = torch.int8).nonzero(as_tuple = False)
    for seq in pred_mask]
    ent_spans = [
        torch.split(T, 2) if len(T) >= 2 else [torch.zeros(2,1)] for T in ent_spans
    ]
    ent_spans =[
        [(T[0].item(), T[1].item()) if len(T) == 2 else (T[0].item(), row_lens[i].item()) for T in spans]
    for i, spans in enumerate(ent_spans)]
    delete_tokens = list(tokenizer.special_tokens_map.values())
    ent_spans = [
        [T for T in spans if T[0] < T[1] and T[1] <= row_lens[i] and not any(tok in delete_tokens or re.search(r"\w+", tok) is None for tok in encoded_inputs.encodings[i].tokens[T[0]:T[1]])]
    for i,spans in enumerate(ent_spans)]

    span_combinations = [
        list(combinations(spans, 2))
    for spans in ent_spans]
    # span_combinations.shape = (batch_size, N_combinations(len(spans), 2))

    # Eliminate documents with no valid span combinations
    valid_documents = [i for i,comb in enumerate(span_combinations) if comb]
    if len(valid_documents) == 0:
        return None, None, None, None

    selected_inputs = encoded_inputs.input_ids[valid_documents]
    attn_mask = attn_mask[valid_documents]
    row_lens = row_lens[valid_documents]
    predictions = predictions[valid_documents]
    pred_mask = pred_mask[valid_documents]
    raw_predictions = raw_predictions[valid_documents]
    ent_spans = [ent_spans[i] for i in valid_documents]
    span_combinations = [span_combinations[i] for i in valid_documents]

    first_order = [
        ent_span_list + span_combo_list
    for ent_span_list, span_combo_list in zip(ent_spans, span_combinations)]

    # Re-ordering index for later
    new_order = list(chain(*[
        [
            sum(map(len, first_order[:doc_n])) + first_order[doc_n].index(s),
            sum(map(len, first_order[:doc_n])) + first_order[doc_n].index(o),
            sum(map(len, first_order[:doc_n])) + first_order[doc_n].index((s,o))
        ]
    for doc_n,span_combo_list in enumerate(span_combinations) for s,o in span_combo_list]))

    # Expand the batch so that for each input sentence there are N_entities + N_pairs inputs: one with each entity masked, and one with both entities masked
    # The order of everything is, for each document: e_1, e_2, ..., e_N, (e_1, e_2), (e_1, e_3), ..., (e_N-1, e_N)
    expanded_batch = torch.cat([
        torch.stack(
            [
                torch.where(torch.logical_and(torch.arange(encoding.shape[0]) >= s[0], torch.arange(encoding.shape[0]) < s[1]), tokenizer.mask_token_id, encoding)
            for s in ent_span_list] +
            [
                torch.where(torch.logical_or(
                    torch.logical_and(torch.arange(encoding.shape[0]) >= s[0], torch.arange(encoding.shape[0]) < s[1]), 
                    torch.logical_and(torch.arange(encoding.shape[0]) >= o[0], torch.arange(encoding.shape[0]) < o[1])
                ), tokenizer.mask_token_id, encoding)
            for s,o in span_combo_list]
        )
        if span_combo_list else 
        torch.empty(0, max_len)
    for encoding, ent_span_list, span_combo_list in zip(selected_inputs.to("cpu"), ent_spans, span_combinations)])
    # expanded_batch.shape = (batch_size*(N_entities + N_pairs), max_len)

    if expanded_batch.shape[0] == 0:
        raise ValueError("No valid span combinations found")

    # Get the ids of the tokens that are perturbed for each input sentence
    eq_class_emb_ids = list(chain(*[
        [
            torch.cat([torch.randint(ent[0], ent[1], (min(ent[1]-ent[0], max_tokens_per_entity),)).reshape(-1,1) for ent in ent_span_list if ent != current_ent])
        for current_ent in ent_span_list] +
        [
            torch.cat([torch.randint(s[0], s[1], (min(s[1]-s[0], max_tokens_per_entity),)).reshape(-1,1), torch.randint(o[0], o[1], (min(o[1]-o[0], max_tokens_per_entity),)).reshape(-1,1)])
        for s,o in span_combo_list]
    for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)]))
    eq_class_emb_ids = torch.nn.utils.rnn.pad_sequence(eq_class_emb_ids, batch_first = True, padding_value = -1).squeeze(-1)
    # eq_class_emb_ids.shape = (batch_size*(N_entities + N_pairs), max_perturbed_entity_spans_length)

    pred_ids = list(chain(*[
        [
            torch.randint(ent[0], ent[1], (min(ent[1]-ent[0], max_tokens_per_entity),)).reshape(-1,1)
        for ent in ent_span_list] +
        [
            torch.cat([torch.randint(s[0], s[1], (min(s[1]-s[0], max_tokens_per_entity),)).reshape(-1,1), torch.randint(o[0], o[1], (min(o[1]-o[0], max_tokens_per_entity),)).reshape(-1,1)])
        for s,o in span_combo_list]
    for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)]))
    pred_ids = torch.nn.utils.rnn.pad_sequence(pred_ids, batch_first = True, padding_value = -1).squeeze(-1)
    # pred_ids.shape = (batch_size*3*N_combinations(len(spans), 2), max_perturbed_entity_spans_length)

    separators = list(chain(*[
        [
            tuple(min(ent[1] - ent[0], max_tokens_per_entity) for ent in ent_span_list)
        ]*len(ent_span_list) +
        [
            (min(s[1] - s[0], max_tokens_per_entity),)
        for s,o in span_combo_list]
    for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)]))

    # We need to select the vocabulary in order to avoid carrying the whole vocabulary into Jacobians
    repetitions = torch.as_tensor([len(ent_span_list) + len(span_combo_list) for ent_span_list, span_combo_list in zip(ent_spans, span_combinations)])
    repeated_attn_mask = attn_mask.repeat_interleave(repetitions, dim = 0)

    assert expanded_batch.shape[0] == eq_class_emb_ids.shape[0] == pred_ids.shape[0] == repeated_attn_mask.shape[0]

    del model
    torch.cuda.empty_cache()
    output_only_model = OutputOnlyModel(mlm_model.to(device))

    # Firdt prediction to select the vocabulary
    token_predictions = output_only_model.model(selected_inputs.to(device)).logits.cpu()
    selected_vocab = token_predictions.topk(k = max_retained_vocab, dim = -1).indices.repeat_interleave(repetitions, dim = 0)

    # Initialize dataset and dataloader
    local_dataset = TensorDataset(expanded_batch, eq_class_emb_ids, pred_ids, repeated_attn_mask, selected_vocab)
    dataloader = DataLoader(local_dataset, batch_size = 3, shuffle = False, pin_memory = True, num_workers = 2)

    final_outputs = {
        "traces": [], 
        "eigenvectors": [], 
        "pred_probas": [],
        "separators": separators
    }

    del selected_inputs, predictions, raw_predictions, token_predictions
    torch.cuda.empty_cache()
    gc.collect()

    for minibatch in tqdm(dataloader):

        minibatch_inputs, minibatch_eq_class_emb_ids, minibatch_pred_ids, minibatch_attn_mask, minibatch_selected_vocab = minibatch
        minibatch_inputs = minibatch_inputs.to(device, dtype = torch.int32)
        minibatch_eq_class_emb_ids = [
            row[row >= 0].unique().tolist()
        for row in minibatch_eq_class_emb_ids]
        minibatch_pred_ids = [
            row[row >= 0].unique().tolist()
        for row in minibatch_pred_ids]
        minibatch_attn_mask = minibatch_attn_mask.to(device, dtype = torch.bool)
        minibatch_selected_vocab = minibatch_selected_vocab.to(device, dtype = torch.int32)

        if hasattr(output_only_model.model.base_model, "embeddings"):
            input_embeddings = output_only_model.model.base_model.embeddings(minibatch_inputs)
        elif hasattr(output_only_model.model.base_model, "embed_tokens"):
            input_embeddings = output_only_model.model.base_model.embed_tokens(minibatch_inputs)
        else:
            raise AttributeError("Model has neither embeddings nor embed_tokens")
        # input_embeddings.shape = (batch_size*3*N_combinations(len(spans), 2), max_len, embedding_size)

        g = torch.eye(selected_vocab.shape[-1], device = input_embeddings.device, dtype = input_embeddings.dtype)
        g = g.unsqueeze(0).unsqueeze(0).expand(input_embeddings.shape[0], max(map(len, minibatch_pred_ids))*max(map(len, minibatch_eq_class_emb_ids)), -1, -1)

        pullback_traces, diverging_eigenvectors, predictions = pullback(
            input_embeddings,
            g = g,
            model = output_only_model,
            eq_class_emb_ids = minibatch_eq_class_emb_ids,
            pred_id = minibatch_pred_ids,
            select = minibatch_selected_vocab,
            attention_mask = minibatch_attn_mask,
            return_trace = True,
            return_predictions = True,
            min_num_eigenvectors = min_num_eigenvectors,
            max_num_eigenvectors = max_num_eigenvectors
        )
        tqdm.write(f"Pullback traces shape: {pullback_traces.shape}")
        tqdm.write(f"Diverging eigenvectors shape: {diverging_eigenvectors.shape}")
        # For eigenvectors, we expect dimensions 1 and 3 to change at each minibatch, so we set a minimum number of eigenvectors to take
        if diverging_eigenvectors.shape[-1] >= min_num_eigenvectors:
            diverging_eigenvectors = diverging_eigenvectors[...,:min_num_eigenvectors]
        else:
            diverging_eigenvectors = torch.cat([
                diverging_eigenvectors,
                torch.zeros(diverging_eigenvectors.shape[0], diverging_eigenvectors.shape[1], diverging_eigenvectors.shape[2], min_num_eigenvectors - diverging_eigenvectors.shape[-1], device = diverging_eigenvectors.device, dtype = diverging_eigenvectors.dtype)
            ], dim = -1)

        final_outputs["traces"].extend([T for T in pullback_traces.cpu()])
        final_outputs["eigenvectors"].extend([T for T in diverging_eigenvectors.cpu()])
        final_outputs["pred_probas"].extend([T for T in predictions.cpu()])

        del minibatch_inputs, minibatch_eq_class_emb_ids, minibatch_pred_ids, minibatch_attn_mask, minibatch_selected_vocab, input_embeddings, g
        torch.cuda.empty_cache()
        gc.collect()

    # Outputs must be of shape (len(batch)*N_combinations(len(spans), 2), 3, ...), where the last dimensions are:
    # - (max_len_entities, max_len_entities) for traces;
    # - (max_len_entities, max_len_entities, embedding_size, max_num_eignevectors) for eigenvectors;
    # - (max_len_entities, selected_vocab_size) for pred_probas;
    # - (1) for separators;

    final_outputs = {
        k:torch.stack(
            pad_traces_and_eigenvectors(v)[new_order].tensor_split(3) if isinstance(v[0], torch.Tensor) else torch.nn.utils.rnn.pad_sequence([torch.as_tensor(tup).reshape(-1,1) for tup in v], batch_first=True)[new_order].tensor_split(3), 
            dim = 1
        )
    for k,v in final_outputs.items()}

    return final_outputs, span_combinations, ent_spans, valid_documents


def arnold_gokhale_denominator(P_i_cond_j: torch.Tensor, P_j_cond_i: torch.Tensor, tol: float = 1e-2, max_iter: int = 1000):
    """From 'Distributions most nearly compatible with given families of conditional distributions' by Arnold and Gokhale (1998)"""

    input_shapes = (P_i_cond_j.shape, P_j_cond_i.shape)
    # Input tensor can have multiple batch_shapes, for example if they represent multiple tokens in the same entity, which need to be treated separately and re-aggregated later

    if P_i_cond_j.dim() > 3 or (P_i_cond_j.dim() == 3 and P_i_cond_j.shape[-1] != 1):
        last_shape_not_one = next(s for s in P_i_cond_j.shape[::-1] if s != 1)
        P_i_cond_j = P_i_cond_j.reshape(-1, last_shape_not_one, 1)
    if P_j_cond_i.dim() > 3 or (P_j_cond_i.dim() == 3 and P_j_cond_i.shape[-1] != 1):
        last_shape_not_one = next(s for s in P_j_cond_i.shape[::-1] if s != 1)
        P_j_cond_i = P_j_cond_i.reshape(-1, 1, last_shape_not_one)

    batch_size = P_i_cond_j.shape[0] if P_i_cond_j.squeeze().dim() == 2 else 1
    P_i_cond_j = P_i_cond_j.reshape(batch_size, -1, 1)
    P_j_cond_i = P_j_cond_i.reshape(batch_size, 1, -1)

    Xi = torch.randn_like(P_i_cond_j).softmax(dim = -1)
    Xj = torch.randn_like(P_j_cond_i).softmax(dim = -2)

    Xij = Xi * Xj
    # shape = (batch_size, domain_size, domain_size)
    delta = tol + 1

    for t in range(max_iter):

        inv_sum = 1 / (Xi + EPS) + 1 / (Xj + EPS)
        num = (P_i_cond_j + P_j_cond_i) / inv_sum
        den = num.sum() + EPS
        new_Xij = num / den
        delta = torch.norm(new_Xij - Xij, p = 1)
        Z = den * inv_sum

        if delta < tol:
            break

        Xij = deepcopy(new_Xij)
        Xi = Xij.sum(dim = -1, keepdim = True)
        Xj = Xij.sum(dim = -2, keepdim = True)

    back_to_input_shape = list(input_shapes[0][:-1]) + [Z.shape[-1]] if input_shapes[0][-1] == 1 else list(input_shapes[0]) + [Z.shape[-1]]
    return Z.reshape(back_to_input_shape)


def and_operation(x: torch.Tensor, dim: int = -1):
    """Exp-sum-log operation to avoid that too small values become 0 in an approximation of minimum or product."""
    
    max_value = torch.finfo(x.dtype).max
    mask = torch.logical_and(torch.isfinite(x), x > 0)
    x = torch.where(mask, x, torch.ones_like(x) * EPS)
    output = torch.log(x + EPS).sum(dim = dim).exp()
    output = torch.clamp(output, min = -max_value, max = max_value)
    output = torch.where(mask.sum(dim = dim) > 0, output, torch.zeros_like(output))

    nonfinite_values = torch.logical_or(torch.isnan(output), torch.isinf(output)).sum().item()
    if nonfinite_values >= output.numel() / 2:
        raise ValueError("Too many non-finite values in and_operation")
    elif nonfinite_values > 0:
        print(f"Warning: {nonfinite_values} non-finite values in and_operation. Turning them into 0.")
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
        
    return output


@torch.no_grad()
@timing_decorator
def compute_mutual_information(traces: torch.Tensor, pred_probas: torch.Tensor, separators: torch.Tensor, p_minkowski: float = 2):

    # Inputs of shape (len(batch)*N_combinations(len(spans), 2), 3, ...)
    predictions, pred_indices = pred_probas.max(dim = -1)
    # predictions.shape = (len(batch)*N_combinations(len(spans), 2), 3, max_len_entities)

    AG_dens = arnold_gokhale_denominator(pred_probas[:,0], pred_probas[:,1])
    # AG_dens.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities, selected_vocab_size, selected_vocab_size)
    AG_dens = torch.gather(AG_dens, dim = -1, index = pred_indices[:,1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, AG_dens.shape[-2], -1))
    AG_dens = torch.gather(AG_dens, dim = -2, index = pred_indices[:,0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, AG_dens.shape[-1]))
    AG_dens = AG_dens.squeeze(-1)
    # AG_dens.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities, 1)

    num_per_token_traces = predictions[:,0, :traces.shape[2]].unsqueeze(-1) * traces[:,1, :traces.shape[2]] + predictions[:,1,:traces.shape[2]].unsqueeze(-1) * traces[:,0, :traces.shape[2]]
    num_per_token_traces = num_per_token_traces / (AG_dens[:,:num_per_token_traces.shape[1]] + EPS)

    predictions = predictions.squeeze(-1)
    sep_lengths = separators.diff(dim = -1)[...,::2]
    positions_of_predictions = torch.arange(predictions.shape[2]).reshape(1, -1).expand(predictions.shape[0], -1)
    mask_Pi = torch.where(positions_of_predictions < sep_lengths[:,2,[0]] + 1, 1, 0)
    mask_Pj = torch.where(positions_of_predictions >= sep_lengths[:,2,[0]] + 1, 1, 0)

    subj_positions_in_traces = torch.arange(traces.shape[2]).reshape(1, -1, 1).expand(traces.shape[0], -1, traces.shape[3])
    obj_positions_in_traces = torch.arange(traces.shape[3]).reshape(1, 1, -1).expand(traces.shape[0], traces.shape[2], -1)
    mask_Tij = torch.where(
        torch.logical_or(
            torch.logical_and(
                subj_positions_in_traces < sep_lengths[:,2,[0]].unsqueeze(-1) + 1,
                torch.logical_and(
                    obj_positions_in_traces >= separators[:,2,[2]].unsqueeze(-1),
                    obj_positions_in_traces < separators[:,2,[3]].unsqueeze(-1) + 1
                )
            ),
            torch.logical_and(
                subj_positions_in_traces >= sep_lengths[:,2,[0]].unsqueeze(-1) + 1,
                torch.logical_and(
                    obj_positions_in_traces >= separators[:,2,[0]].unsqueeze(-1),
                    obj_positions_in_traces < separators[:,2,[1]].unsqueeze(-1) + 1
                )
            )
        ), 1, 0)
    
    den_Pi = predictions[:,2] * mask_Pi
    den_Pj = predictions[:,2] * mask_Pj
    den_Tij = traces[:,2] * mask_Tij
    den_per_token_traces = torch.stack([
        den_Tij * den_Pi[:,:den_Tij.shape[1]].unsqueeze(-1), 
        den_Tij * den_Pj[:,:den_Tij.shape[1]].unsqueeze(-1)
        ],
        dim = -1
    )
    den_per_token_traces = and_operation(den_per_token_traces, dim = -1)
    # shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities, max_len_entities)

    # Both numerator and denominator are non-negative, since they are the sum of products between probabilities and traces of positive semi-definite matrices
    agg_num = and_operation(num_per_token_traces.pow(p_minkowski).sum(dim = -1).pow(1/p_minkowski), dim = -1) # equivalent in Van Krieken's fuzzy logic to: forall obj (exists subj Influences(subj, obj))
    agg_den = and_operation(den_per_token_traces.pow(p_minkowski).sum(dim = -1).pow(1/p_minkowski), dim = -1) # equivalent in Van Krieken's fuzzy logic to: forall obj (exists subj Influences(subj, obj))
    # shape = (len(batch)*N_combinations(len(spans), 2))

    return torch.log(EPS + agg_num / (agg_den + EPS)) * agg_num


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
        # Generate all combinations of 2 indices
        combos = combinations(range(num_indices), 2)
        
        doc_result = sum([[combo, combo, combo] for combo in combos], [])
        doc_result = list(map(list, doc_result))
        
        result.extend(doc_result)
    
    return result


@timing_decorator
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_dataset("xiaobendanyn/tacred", split = "test")
    df = pd.DataFrame([json.loads(d['text']) for d in data]).map(lambda x: x['name'] if isinstance(x, dict) else " ".join(x) if isinstance(x, list) else x)
    df.rename(columns = {"token": "input", "relation": "label"}, inplace = True)

    model_name = "numind/NuNER-v2.0"
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    batch_size = 16
    max_num_eigenvectors = model.config.embedding_size // 2 if hasattr(model.config, "embedding_size") else model.config.hidden_size // 2
    os.makedirs("../results", exist_ok = True)

    ex = df["input"].tolist()
    full_results = []
    running_index = 0
    full_inputs = tokenizer(ex, padding = "max_length", add_special_tokens = False, max_length = model.config.max_position_embeddings - 2 if hasattr(model.config, "max_position_embeddings") else model.config.n_ctx - 2, truncation=True, return_tensors = "pt")

    mlm_model = train_lm_head(model_name, device, mask_id = tokenizer.mask_token_id, n_epochs = 10, inputs = full_inputs.input_ids, attn_mask = full_inputs.attention_mask)

    for batch in batch_generator(ex, batch_size = batch_size):
        pullback_dict, entity_span_combinations, entity_spans, valid_document_numbers = forward_backward_pass(batch, model, tokenizer, mlm_model, max_num_eigenvectors = max_num_eigenvectors)
        if any(res is None for res in [pullback_dict, entity_span_combinations, entity_spans, valid_document_numbers]):
            print("No valid results found for batch", flush = True)
            running_index += batch_size
            continue

        # Select the correct separators (s_1,s_2,o_1,o_2) for each sentence
        slice_indices = torch.as_tensor([x for x in get_combination_indices_with_repeats(entity_spans)])
        slice_indices = torch.cat([slice_indices, slice_indices+1], dim = -1)[:, [0,2,1,3]]
        separators = torch.cat([torch.zeros_like(pullback_dict["separators"].squeeze(-1)[...,:1]), pullback_dict["separators"].squeeze(-1)], dim = -1)
        separators = torch.gather(
            separators.cumsum(dim = -1),
            dim = -1,
            index = slice_indices.reshape(-1,3,4).to(pullback_dict["separators"].device, dtype = torch.int32)
        )

        mi = compute_mutual_information(pullback_dict["traces"], pullback_dict["pred_probas"], separators)
        print(mi.shape)

        # Normalize scores to make them intelligible
        current_pos = 0
        for i,sent_span_combinations in entity_span_combinations:
            k = len(sent_span_combinations)
            mi[current_pos:current_pos+k] = (mi[current_pos:current_pos+k] - mi[current_pos:current_pos+k].min()) / (mi[current_pos:current_pos+k].max() - mi[current_pos:current_pos+k].min())
            current_pos += k

            entity_pairs_scores = [
                {
                    "e_i_idx": list(span_combo[0]),
                    "e_j_idx": list(span_combo[1]),
                    "e_i_tokens": " ".join(tokenizer.tokenize(batch[i], add_special_tokens = False)[span_combo[0][0]:span_combo[0][1]]),
                    "e_j_tokens": " ".join(tokenizer.tokenize(batch[i], add_special_tokens = False)[span_combo[1][0]:span_combo[1][1]]),
                    "score": mi[current_pos + j].to(dtype = torch.float64).item(),
                    "original_text": batch[i],
                    "doc_number": running_index + valid_document_numbers[i]
                }
            for j,span_combo in enumerate(sent_span_combinations)]

            full_results.extend(entity_pairs_scores)

        running_index += batch_size
    
        with open(f"../results/tacred_scores_{model_name.split('/')[-1]}.json", "w") as f:
            json.dump(full_results, f)

    print(f"Saved results to ../results/tacred_scores_{model_name.split('/')[-1]}.json", flush = True)


if __name__ == "__main__":
    main()