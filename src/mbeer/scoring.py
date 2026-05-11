"""Main script"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from difflib import SequenceMatcher
import gc
import itertools
from itertools import chain, combinations
import json
import math
import os
import re
from typing import List, Tuple
import warnings

from datasets import load_dataset
from huggingface_hub import HfApi
import pandas as pd
from scipy.stats import chi2
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from mbeer.baseline_functions import (
    mean_integrated_directional_gradients,
    shapley_interaction_quantification,
)
from mbeer.mbrd import RelationDisambiguation

hf_api = HfApi()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EPS = 2*torch.finfo(torch.float32).eps
torch.set_default_dtype(torch.float32)
_WORD_RE = re.compile(r"\w+")

def _mask_spans(encoding: torch.Tensor, spans: List[Tuple[int, int]], mask_id: int) -> torch.Tensor:
    """Clone `encoding` and overwrite all positions in any of `spans` with `mask_id`."""
    out = encoding.clone()
    for s in spans:
        out[s[0]:s[1]] = mask_id
    return out

# Handle both relative and absolute imports
try:
    from .pullback import (OutputOnlyModel,
                           compute_pred_ids_and_eq_class_emb_ids, pullback)
    from .utils import batch_generator, timing_decorator
except ImportError:
    # If relative import fails, try absolute import
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from mbeer.pullback import OutputOnlyModel, compute_pred_ids_and_eq_class_emb_ids, pullback
    from mbeer.utils import batch_generator, timing_decorator


def _resolve_torch_device(d) -> torch.device:
    """Canonical CUDA device for dict keys and set_device (cuda -> cuda:0)."""
    
    x = torch.device(d) if d is not None else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
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
            raise ValueError("All tensors must have the same number of dimensions or be reshapeable to the same number of dimensions")

    if dim < 3:
        max_last_dim = max(T.shape[-1] for T in L)
        padded_tensors = [T.unsqueeze(-1) if T.dim() == 1 else torch.nn.functional.pad(T, (0,max_last_dim-T.shape[1])) for T in L]
        return torch.nn.utils.rnn.pad_sequence(padded_tensors, batch_first = True, padding_value = 0)

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

    output = torch.cat(L)

    if torch.isnan(output).any():
        warnings.warn("NaN values in output")

    return output
    

def train_lm_head(model_name: str, device: torch.device, mask_id: int, n_epochs: int = 10, inputs: torch.Tensor = None, attn_mask: torch.Tensor = None, encoder_name: str = "encoder", patience: int = None, tol: float = 1e-6):

    parent_model_info = hf_api.model_info(model_name)
    parent_model_name = parent_model_info.card_data.get("base_model", None)
    if isinstance(parent_model_name, list):
        parent_model_name = parent_model_name[0]
    ignore_modules = ["dropout", "activation", "act_fn"]

    if patience is None:
        patience = n_epochs // 10

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    masked_inputs = torch.where(torch.rand_like(inputs.to(dtype = torch.float32)) < 0.15, mask_id, inputs)

    if parent_model_name is not None:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(parent_model_name.strip("-.][")).to(device, dtype = torch.float32), encoder_name = encoder_name)

    else:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(model_name).to(device, dtype = torch.float32), encoder_name = encoder_name)
        are_weights_initialized ={k:all(torch.all(p.diff() < 1e-6) for p in v.parameters()) for k,v in output_only_model.model.named_modules() if hasattr(v, "parameters") and all(ign not in k for ign in ignore_modules)}
        uninitialized_modules = [k for k,v in are_weights_initialized.items() if v]
        uninitialized_parameters = [p for v in uninitialized_modules for p in output_only_model.model.get_submodule(v).parameters()]

        mask_token_id = tokenizer.mask_token_id

        if len(uninitialized_modules) > 0:

            modules_path = os.path.join("models", model_name.split("/")[-1])

            if os.path.exists(modules_path) and len(os.listdir(modules_path)) > 0:
                for k in uninitialized_modules:
                    try:
                        output_only_model.model.get_submodule(k).load_state_dict(torch.load(os.path.join(modules_path, f"{k}.pt")))
                    except FileNotFoundError:
                        warnings.warn(f"Could not load weights for module {k} from {modules_path}")

                return output_only_model

            os.makedirs(os.path.join("models", model_name.split("/")[-1]), exist_ok = True)

            output_only_model.model.requires_grad_(False)
            for p in uninitialized_parameters:
                p.requires_grad_(True)

            with torch.enable_grad():
                for module in uninitialized_modules:
                    output_only_model.model.get_submodule(module).train()

                loss_fn = torch.nn.BCEWithLogitsLoss(reduction = "mean")
                optimizer = torch.optim.LBFGS(uninitialized_parameters, lr = 1e-3, history_size = max(n_epochs // 10, 1), max_iter = 1)

                dataset = TensorDataset(masked_inputs, inputs, attn_mask)
                dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, pin_memory = True, num_workers = 2)

                def closure(logits, targets):
                    optimizer.zero_grad()
                    vocab_size = logits.shape[-1]
                    loss = loss_fn(
                        logits.view(-1, vocab_size), 
                        F.one_hot(targets.view(-1), vocab_size).to(logits.device, dtype = logits.dtype)
                    )
                    loss.backward()
                    return loss

                loss_sequence = torch.ones(patience, device = device) * float("inf")

                for epoch in trange(n_epochs, desc = "Training LM head"):
                    epoch_loss = []
                    for minibatch in dataloader:
                        selected_inputs, selected_targets, selected_attn_mask = minibatch
                        selected_inputs = selected_inputs.to(device)
                        selected_targets = selected_targets.to(device)
                        selected_attn_mask = selected_attn_mask.to(device)
                        y = output_only_model.model(selected_inputs, attention_mask=selected_attn_mask).logits
                        masked_tokens_mask = torch.repeat_interleave(selected_inputs.reshape(selected_inputs.shape[0], -1, 1) == mask_token_id, y.shape[-1], dim = -1)
                        masked_logits = y[masked_tokens_mask].reshape(-1, y.shape[-1])
                        masked_targets = selected_targets[masked_tokens_mask[...,0]].reshape(-1)
                        current_loss = optimizer.step(lambda: closure(masked_logits, masked_targets))
                        epoch_loss.append(current_loss.item())
                    epoch_loss = torch.as_tensor(epoch_loss).mean().item()
                    loss_sequence[epoch % patience] = epoch_loss
                    if torch.all(loss_sequence <= max(epoch_loss, tol)):
                        break
                    print(f"Epoch {epoch}, loss: {epoch_loss:.4f}")

            optimizer.zero_grad(set_to_none = True)
            del optimizer, loss_fn

            output_only_model.model.requires_grad_(False)

            for module in uninitialized_modules:
                output_only_model.model.get_submodule(module).eval()
                torch.save(output_only_model.model.get_submodule(module).state_dict(), os.path.join("models", model_name.split("/")[-1], f"{module}.pt"))

    return output_only_model


@torch.no_grad()
@timing_decorator
def forward_backward_pass(batch: List[str], model: AutoModelForTokenClassification, tokenizer: AutoTokenizer, mlm_model: AutoModelForMaskedLM, gold_spans: List[List[Tuple[int, int]]] = None, min_num_eigenvectors: int = 64, max_num_eigenvectors: int = None, max_tokens_per_entity: int = 4, max_retained_vocab: int = 30, minibatch_size: int = 3, encoder_name: str = "encoder"):

    max_len = model.config.max_position_embeddings - 2 if hasattr(model.config, "max_position_embeddings") else model.config.n_ctx - 2
    pred_to_mask = {k:int(v.startswith(('B','U'))) + 2*int(v.startswith(('I','L')) and not v.endswith('0')) for k,v in model.config.id2label.items()} if hasattr(model.config, "id2label") else {}
    local_device = model.device
    if local_device.type == "cuda":
        torch.cuda.set_device(local_device)

    encoded_inputs = tokenizer(batch, padding = "max_length", add_special_tokens = False, max_length = max_len, truncation=True, return_tensors = "pt").to(local_device)
    attn_mask = encoded_inputs.attention_mask
    row_lens = attn_mask.sum(dim = 1)
    num_labels = len(pred_to_mask) if pred_to_mask else 2
    # Threshold from power of Chi-Squared test for uniformity of distribution
    probit_threshold = 1 / num_labels + math.sqrt(chi2.sf(0.95, num_labels - 1) / (2*num_labels))
    probit_threshold = min(probit_threshold, 0.5)

    logit_threshold = torch.nn.Sequential(torch.nn.Softmax(dim = -1), torch.nn.Threshold(probit_threshold, 0))

    raw_predictions = model(**encoded_inputs).logits
    predictions = logit_threshold(raw_predictions * attn_mask.unsqueeze(-1).to(raw_predictions.device)).cpu()
    if predictions.shape[-1] > 1:
        predictions = predictions.argmax(dim = -1)
    else:
        predictions = predictions.round().to(dtype = torch.int8).squeeze(-1)
    # predictions.shape = (batch_size, max_len)

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
    # An entity starts at any B/U, or at an I/L that follows an O (defensive against malformed BIO).
    is_start = (pred_mask == 1) | ((pred_mask == 2) & (prev_mask == 0))
    # An entity ends (exclusive) where the previous token was inside an entity and the current token is O or starts a new entity.
    is_end = (prev_mask != 0) & ((pred_mask == 0) | (pred_mask == 1))

    ent_spans = []
    for i in range(pred_mask.shape[0]):
        starts = is_start[i].nonzero(as_tuple=True)[0].tolist()
        ends = is_end[i].nonzero(as_tuple=True)[0].tolist()
        # Close any entity that runs up to the last valid token (no following O to mark its end).
        if len(starts) > len(ends):
            ends.append(row_lens[i].item())
        ent_spans.append(list(zip(starts, ends)) + gold_spans[i])

    delete_tokens = set(tokenizer.special_tokens_map.values())
    row_lens_list = row_lens.tolist()
    batch_tokens_per_row = [enc.tokens for enc in encoded_inputs.encodings]
    ent_spans = [
        [T for T in spans if T[0] < T[1] and T[1] <= row_lens_list[i] and not any(tok in delete_tokens or _WORD_RE.search(tok) is None for tok in batch_tokens_per_row[i][T[0]:T[1]])]
    for i,spans in enumerate(ent_spans)]

    span_combinations = [
        list(combinations(spans, 2)) if len(spans) > 1 else []
    for spans in ent_spans]
    # span_combinations.shape = (batch_size, N_combinations(len(spans), 2))

    # Correct possible issues with empty entity spans
    for ent_span_list, span_combo_list in zip(ent_spans, span_combinations):
        for ent in ent_span_list:
            if ent[1] - ent[0] <= 0:
                ent_span_list.remove(ent)

        for s,o in span_combo_list:
            if s[1] - s[0] <= 0 or o[1] - o[0] <= 0:
                span_combo_list.remove((s,o))
            if s not in ent_span_list:
                ent_span_list.append(s)
            if o not in ent_span_list:
                ent_span_list.append(o)    

    # Eliminate documents with no valid span combinations
    valid_documents = [i for i,combos in enumerate(span_combinations) if combos]
    if len(valid_documents) == 0:
        warnings.warn("No valid span combinations found")
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

    # Re-ordering index for later. Precompute per-document offsets and
    # item-to-index maps so each (s, o, (s,o)) lookup is O(1) instead of
    # the original O(N) list-prefix sum + O(N) list.index per pair.
    doc_offsets = [0]
    for doc in first_order:
        doc_offsets.append(doc_offsets[-1] + len(doc))
    doc_index_map = [{item: idx for idx, item in enumerate(doc)} for doc in first_order]

    new_order: List[int] = []
    for doc_n, span_combo_list in enumerate(span_combinations):
        base = doc_offsets[doc_n]
        idx_map = doc_index_map[doc_n]
        for s, o in span_combo_list:
            new_order.append(base + idx_map[s])
            new_order.append(base + idx_map[o])
            new_order.append(base + idx_map[(s, o)])

    # Expand the batch so that for each input sentence there are N_entities + N_pairs inputs: one with each entity masked, and one with both entities masked
    # The order of everything is, for each document: e_1, e_2, ..., e_N, (e_1, e_2), (e_1, e_3), ..., (e_N-1, e_N)
    mask_tok_id = tokenizer.mask_token_id
    selected_inputs_cpu = selected_inputs.to("cpu")
    expanded_batch = torch.cat([
        torch.stack(
            [_mask_spans(encoding, [s], mask_tok_id) for s in ent_span_list] +
            [_mask_spans(encoding, [s, o], mask_tok_id) for s, o in span_combo_list]
        )
        if span_combo_list else
        torch.empty(0, max_len, dtype=encoding.dtype)
    for encoding, ent_span_list, span_combo_list in zip(selected_inputs_cpu, ent_spans, span_combinations)])
    # expanded_batch.shape = (batch_size*(N_entities + N_pairs), max_len)

    if expanded_batch.shape[0] == 0:
        raise ValueError("No valid span combinations found")

    # Cache sampled positions per unique entity span so identical spans
    # (the common case across pair combinations within a document) share
    # the same tensor instead of rebuilding it.
    pred_ids, eq_class_emb_ids = compute_pred_ids_and_eq_class_emb_ids(span_combinations, ent_spans, max_tokens_per_entity)

    separators = list(chain(*[
        [
            tuple(max_tokens_per_entity//2 for ent in ent_span_list)
        ]*len(ent_span_list) +
        [
            (max_tokens_per_entity//2,)
        for s,o in span_combo_list]
    for span_combo_list, ent_span_list in zip(span_combinations, ent_spans)]))

    # We need to select the vocabulary in order to avoid carrying the whole vocabulary into Jacobians
    repetitions = torch.as_tensor([len(ent_span_list) + len(span_combo_list) for ent_span_list, span_combo_list in zip(ent_spans, span_combinations)])
    repeated_attn_mask = attn_mask.repeat_interleave(repetitions.to(attn_mask.device), dim = 0)

    assert expanded_batch.shape[0] == eq_class_emb_ids.shape[0] == pred_ids.shape[0] == repeated_attn_mask.shape[0]

    del model
    if isinstance(mlm_model, OutputOnlyModel):
        output_only_model = mlm_model.to(local_device)
    else:
        output_only_model = OutputOnlyModel(mlm_model.to(local_device), encoder_name = encoder_name)

    # Firdt prediction to select the vocabulary
    token_predictions = output_only_model.model(selected_inputs.to(local_device)).logits.cpu()
    selected_vocab = token_predictions.topk(k = max_retained_vocab, dim = -1).indices.repeat_interleave(repetitions, dim = 0)

    # Reorder into (entity_s, entity_o, pair) triplets so that each
    # minibatch of 3 fed to pullback forms a valid grouping.
    triplet_order = torch.as_tensor(new_order)
    expanded_batch = expanded_batch[triplet_order]
    eq_class_emb_ids = eq_class_emb_ids[triplet_order]
    pred_ids = pred_ids[triplet_order]
    repeated_attn_mask = repeated_attn_mask[triplet_order]
    selected_vocab = selected_vocab[triplet_order]
    separators = [separators[i] for i in new_order]

    # Initialize dataset and dataloader
    local_dataset = TensorDataset(expanded_batch.cpu(), eq_class_emb_ids.cpu(), pred_ids.cpu(), repeated_attn_mask.cpu(), selected_vocab.cpu())
    dataloader = DataLoader(local_dataset, batch_size = minibatch_size, shuffle = False, pin_memory = True, num_workers = 0)

    final_outputs = defaultdict(list)
    final_outputs["separators"].extend(separators)

    del selected_inputs, predictions, raw_predictions, token_predictions
    torch.cuda.empty_cache()
    gc.collect()

    for minibatch in tqdm(dataloader, desc = f"Batch on device {local_device}"):

        minibatch_inputs, minibatch_eq_class_emb_ids, minibatch_pred_ids, minibatch_attn_mask, minibatch_selected_vocab = minibatch
        minibatch_inputs = minibatch_inputs.to(local_device, dtype = torch.int32)
        minibatch_eq_class_emb_ids = [
            row[row >= 0].unique().tolist()
        for row in minibatch_eq_class_emb_ids]
        minibatch_pred_ids = [
            row[row >= 0].unique().tolist()
        for row in minibatch_pred_ids]
        minibatch_attn_mask = minibatch_attn_mask.to(local_device, dtype = torch.bool)
        minibatch_selected_vocab = minibatch_selected_vocab.to(local_device, dtype = torch.int32)

        if hasattr(output_only_model.model.base_model, "embeddings"):
            input_embeddings = output_only_model.model.base_model.embeddings(minibatch_inputs)
        elif hasattr(output_only_model.model.base_model, "embed_tokens"):
            input_embeddings = output_only_model.model.base_model.embed_tokens(minibatch_inputs)
        else:
            raise AttributeError("Model has neither embeddings nor embed_tokens")
        # input_embeddings.shape = (batch_size*3*N_combinations(len(spans), 2), max_len, embedding_size)

        g = torch.eye(selected_vocab.shape[-1], device = input_embeddings.device, dtype = input_embeddings.dtype)
        g = g.unsqueeze(0).unsqueeze(0).expand(input_embeddings.shape[0], max(map(len, minibatch_pred_ids))*max(map(len, minibatch_eq_class_emb_ids)), -1, -1)

        pullback_pdets, pullback_traces, pullback_radii, diverging_eigenvectors, predictions = pullback(
            input_embeddings,
            g = g,
            model = output_only_model,
            eq_class_emb_ids = minibatch_eq_class_emb_ids,
            pred_id = minibatch_pred_ids,
            select = minibatch_selected_vocab,
            attention_mask = minibatch_attn_mask,
            approximated_eigendecomposition = False,
            return_trace = True,
            return_predictions = True,
            min_num_eigenvectors = min_num_eigenvectors,
            max_num_eigenvectors = max_num_eigenvectors
        )
        tqdm.write(f"Pullback pdets shape: {pullback_pdets.shape}")
        tqdm.write(f"Pullback traces shape: {pullback_traces.shape}")
        tqdm.write(f"Pullback radii shape: {pullback_radii.shape}")
        tqdm.write(f"Diverging eigenvectors shape: {diverging_eigenvectors.shape}")
        # For eigenvectors, we expect dimensions 1 and 3 to change at each minibatch, so we set a minimum number of eigenvectors to take
        if diverging_eigenvectors.shape[-1] >= min_num_eigenvectors:
            diverging_eigenvectors = diverging_eigenvectors[...,:min_num_eigenvectors]
        else:
            diverging_eigenvectors = torch.cat([
                diverging_eigenvectors,
                torch.zeros(diverging_eigenvectors.shape[0], diverging_eigenvectors.shape[1], diverging_eigenvectors.shape[2], min_num_eigenvectors - diverging_eigenvectors.shape[-1], device = diverging_eigenvectors.device, dtype = diverging_eigenvectors.dtype)
            ], dim = -1)

        final_outputs["pdets"].extend([T for T in pullback_pdets.cpu()])
        final_outputs["traces"].extend([T for T in pullback_traces.cpu()])
        final_outputs["radii"].extend([T for T in pullback_radii.cpu()])
        final_outputs["eigenvectors"].extend([T for T in diverging_eigenvectors.cpu()])
        final_outputs["pred_probas"].extend([T for T in predictions.cpu()])

        del minibatch_inputs, minibatch_eq_class_emb_ids, minibatch_pred_ids, minibatch_attn_mask, minibatch_selected_vocab, input_embeddings, g
        torch.cuda.empty_cache()
        gc.collect()

    pred_probas = final_outputs.pop("pred_probas")

    final_outputs = {
        k:pad_traces_and_eigenvectors(v) if isinstance(v[0], torch.Tensor) else torch.nn.utils.rnn.pad_sequence([torch.as_tensor(tup).reshape(-1,1) for tup in v], batch_first=True)
    for k,v in final_outputs.items()}

    eq_cap = max_tokens_per_entity * 2
    for key in ["pdets", "traces", "radii"]:
        if key in final_outputs:
            final_outputs[key] = final_outputs[key][:, :, :eq_cap]
    if "eigenvectors" in final_outputs:
        final_outputs["eigenvectors"] = final_outputs["eigenvectors"][:, :, :eq_cap]

    # Data is already in triplet order: entity_s at [::3], entity_o at [1::3], pair at [2::3]
    final_outputs["separators"] = final_outputs["separators"][::3]
    pred_probas = torch.nn.utils.rnn.pad_sequence(pred_probas, batch_first=True)
    final_outputs['cond_probas'] = pred_probas[::3] + pred_probas[1::3]
    final_outputs['mask_probas'] = pred_probas[2::3]

    return final_outputs, span_combinations, ent_spans, valid_documents


def arnold_gokhale_denominator(P_i_cond_j: torch.Tensor, P_j_cond_i: torch.Tensor, vocab_size: int = None, tol: float = 1e-2, max_iter: int = 100):
    """From 'Distributions most nearly compatible with given families of conditional distributions' by Arnold and Gokhale (1998)"""

    if vocab_size is None:
        vocab_size = P_i_cond_j.shape[-1]

    assert P_i_cond_j.shape[-1] == vocab_size == P_j_cond_i.shape[-1]
    input_shapes = (P_i_cond_j.shape, P_j_cond_i.shape)
    assert input_shapes[0][0] == input_shapes[1][0]
    batch_size = input_shapes[0][0]
    # Input tensor can have multiple batch_shapes, for example if they represent multiple tokens in the same entity, which need to be treated separately and re-aggregated later

    P_i_cond_j = P_i_cond_j.reshape(batch_size, -1, 1, vocab_size, 1)
    P_j_cond_i = P_j_cond_i.reshape(batch_size, 1, -1, 1, vocab_size)

    P_i_cond_j = torch.nn.functional.softmax(P_i_cond_j, dim = -2)
    P_j_cond_i = torch.nn.functional.softmax(P_j_cond_i, dim = -1)
    idx_i_cond_j = P_i_cond_j.argmax(dim = -2).unsqueeze(-2)
    idx_j_cond_i = P_j_cond_i.argmax(dim = -1).unsqueeze(-1)

    Xi = torch.randn_like(P_i_cond_j).softmax(dim = -2)
    Xj = torch.randn_like(P_j_cond_i).softmax(dim = -1)

    Xij = Xi * Xj
    # Xij.shape = (batch_size, max_len_entities, max_len_entities, selected_vocab_size, selected_vocab_size)
    delta = tol + 1

    for t in range(max_iter):

        inv_sum = torch.reciprocal(Xi + EPS) + torch.reciprocal(Xj + EPS)
        num = (P_i_cond_j + P_j_cond_i) / inv_sum
        # num.shape = (batch_size, max_len_entities, max_len_entities, selected_vocab_size, selected_vocab_size)
        den = num.sum(dim = -1, keepdim = True).sum(dim = -2, keepdim = True)
        # den.shape = (batch_size, max_len_entities, max_len_entities, 1, 1)
        new_Xij = num / (den + EPS)
        delta = torch.norm(new_Xij - Xij, p = 1)
        Z = den * inv_sum

        if delta.item() < tol:
            break

        # `new_Xij` is a freshly computed tensor on every iteration, so we can
        # rebind directly instead of deep-copying it.
        Xij = new_Xij
        Xi = Xij.sum(dim = -1, keepdim = True)
        Xj = Xij.sum(dim = -2, keepdim = True)

    if idx_i_cond_j.shape[2] < Z.shape[2]:
        idx_i_cond_j = idx_i_cond_j.repeat_interleave(Z.shape[2] // idx_i_cond_j.shape[2], dim = 2)
    if idx_j_cond_i.shape[1] < Z.shape[1]:
        idx_j_cond_i = idx_j_cond_i.repeat_interleave(Z.shape[1] // idx_j_cond_i.shape[1], dim = 1)

    Z = torch.gather(Z, dim = -2, index = idx_i_cond_j.expand(-1,-1,-1,-1, vocab_size))
    Z = torch.gather(Z, dim = -1, index = idx_j_cond_i)

    back_to_input_shape = (input_shapes[0][0], input_shapes[0][1], input_shapes[1][1])
    Z = Z.reshape(*back_to_input_shape)

    if torch.isnan(Z).any():
        warnings.warn("NaN values in Z")

    P_joints = torch.reshape(P_i_cond_j.max(dim = -2).values + P_j_cond_i.max(dim = -1).values, back_to_input_shape) / (Z + EPS)

    return Z, P_joints


def and_operation(x: torch.Tensor, dim: int = -1):
    """Exp-sum-log operation to avoid that too small values become 0 in an approximation of minimum or product."""
    
    max_value = x.max().item()
    mask = torch.logical_and(torch.isfinite(x), x > 0)
    x = torch.where(mask, x, torch.ones_like(x) - EPS) # Avoid 0 values
    output = torch.log(x + EPS).sum(dim = dim).exp()
    output = torch.clamp(output, min = 0, max = max_value)
    output = torch.where(mask.sum(dim = dim) > 0, output, torch.zeros_like(output))

    nonfinite_values = torch.logical_or(torch.isnan(output), torch.isinf(output)).sum().item()
    if nonfinite_values >= output.numel() / 2:
        raise ValueError("Too many non-finite values in and_operation")
    elif nonfinite_values > 0:
        print(f"Warning: {nonfinite_values} non-finite values in and_operation. Turning them into 0.")
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

    if torch.isnan(output).any():
        warnings.warn("NaN values in output")
    return output


@torch.no_grad()
@timing_decorator
def compute_mutual_information(traces: torch.Tensor, pdets: torch.Tensor, radii: torch.Tensor, cond_probas: torch.Tensor, mask_probas: torch.Tensor, separators: torch.Tensor, p_minkowski: float = 2):

    embedding_size = traces.shape[-1]

    # Inputs of shape (len(batch)*N_combinations(len(spans), 2), ...)
    cond_predictions, cond_pred_indices = cond_probas.max(dim = -1)
    mask_predictions, mask_pred_indices = mask_probas.max(dim = -1)
    batch_size, max_len_entities, vocab_size = cond_probas.shape
    # predictions.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities)

    cond_predictions = cond_predictions.squeeze(-1)
    sep_lengths = separators.diff(dim = -1)[...,::2]
    _dev = cond_probas.device
    positions_of_predictions = torch.arange(cond_predictions.shape[-1], device=_dev).reshape(1, -1).expand(cond_predictions.shape[0], -1)
    mask_Pi = torch.where(positions_of_predictions <= sep_lengths[:,[0]], 1, 0).unsqueeze(-1)
    mask_Pj = torch.where(positions_of_predictions > sep_lengths[:,[0]], 1, 0).unsqueeze(-1)

    # cond_probas.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities, selected_vocab_size)
    vocab_Pi = cond_probas * mask_Pi
    vocab_Pj = cond_probas * mask_Pj
    num_Pi = cond_predictions.unsqueeze(-1) * mask_Pi
    num_Pj = cond_predictions.unsqueeze(-1) * mask_Pj
    den_Pi = mask_predictions.unsqueeze(-1) * mask_Pi
    den_Pj = mask_predictions.unsqueeze(-1) * mask_Pj

    Zs, P_joints = arnold_gokhale_denominator(vocab_Pi, vocab_Pj, vocab_size = vocab_size)
    # Zs.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities, max_len_entities)

    agg_trace = and_operation(traces, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    # num_trace.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities)
    agg_pdet = and_operation(pdets, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    # num_pdet.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities)
    agg_radius = and_operation(radii, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    # num_radius.shape = (len(batch)*N_combinations(len(spans), 2), max_len_entities)
    
    den_Pi_agg = and_operation(den_Pi, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    den_Pj_agg = and_operation(den_Pj, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    num_Pi_agg = and_operation(num_Pi, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    num_Pj_agg = and_operation(num_Pj, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    P_joints_agg = and_operation(P_joints, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    Zs_agg = and_operation(Zs, dim = -1).pow(p_minkowski).mean(dim = -1).pow(1/p_minkowski)
    alpha_i = (num_Pi_agg + num_Pj_agg) / (Zs_agg*num_Pj_agg + EPS)
    alpha_j = (num_Pj_agg + num_Pi_agg) / (Zs_agg*num_Pi_agg + EPS)

    internal_log = torch.log((alpha_i*den_Pi_agg + alpha_j*den_Pj_agg + EPS)/(den_Pi_agg**2 + den_Pj_agg**2 + EPS))
    internal_term_i = (Zs_agg - num_Pi_agg**2 - 1)/(num_Pi_agg*num_Pj_agg*(Zs_agg**2) + EPS)
    internal_term_j = (Zs_agg - num_Pj_agg**2 - 1)/(num_Pi_agg*num_Pj_agg*(Zs_agg**2) + EPS)
    Ci_agg = alpha_i**2 * ((internal_log + internal_term_i)**2 - 1)
    Cj_agg = alpha_j**2 * ((internal_log + internal_term_j)**2 - 1)

    smits = 2*math.sqrt(embedding_size*2) * Zs_agg.pow(-2) *(Ci_agg*torch.sqrt(agg_pdet[::2] * agg_radius[::2] / agg_trace[::2]) + Cj_agg*torch.sqrt(agg_pdet[1::2] * agg_radius[1::2] / agg_trace[1::2]))

    if torch.isnan(smits).any():
        warnings.warn("NaN values in smits")
        smits = smits.nan_to_num(0)

    pointwise_mi = torch.log((P_joints_agg + EPS) / (den_Pi_agg * den_Pj_agg + EPS))

    if torch.isnan(pointwise_mi).any():
        warnings.warn("NaN values in pointwise_mi")
        pointwise_mi = pointwise_mi.nan_to_num(0)

    return smits.flatten(), pointwise_mi.flatten()


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
        doc_result = [list(combo) for combo in chain.from_iterable((c, c, c) for c in combos)]

        result.extend(doc_result)

    return result



@timing_decorator
def main(test: bool = False, model_name: str = None, with_mbrd: bool = False, with_baselines: bool = False, devices: List[str] = None):
    
    print(f"Scoring model: {model_name}")
    if devices is not None:
        train_device = torch.device(devices[0])
    else:
        train_device = _resolve_torch_device(None)
        
    training_data = load_dataset("xiaobendanyn/tacred", split = "train")
    data = load_dataset("xiaobendanyn/tacred", split = "test")
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
    
    train_df = pd.DataFrame([json.loads(d['text']) for d in training_data]).map(lambda x: x['name'] if isinstance(x, dict) else " ".join(x) if isinstance(x, list) else x)
    df = pd.DataFrame([json.loads(d['text']) for d in data]).map(lambda x: x['name'] if isinstance(x, dict) else " ".join(x) if isinstance(x, list) else x)
    df.rename(columns = {"token": "input", "relation": "label"}, inplace = True)

    # Tokenize every sentence once and reuse the BatchEncoding's char_to_token
    # mapping, instead of re-tokenizing per row (and per query).
    inputs_list = df['input'].tolist()
    batch_encodings = tokenizer(inputs_list, add_special_tokens=False, padding=False)

    def _find_tokens(idx, query):
        s = inputs_list[idx]
        result = []
        for match in SequenceMatcher(None, s, query).get_matching_blocks():
            char_start = match.a
            char_end = match.a + match.size
            ts = batch_encodings.char_to_token(idx, char_start)
            if ts is None:
                continue
            te = batch_encodings.char_to_token(idx, char_end - 1)
            if te is None:
                continue
            result.append((ts, te + 1))
        return result

    df['h_token_spans'] = [_find_tokens(i, h) for i, h in enumerate(df['h'])]
    df['t_token_spans'] = [_find_tokens(i, t) for i, t in enumerate(df['t'])]
    df['gold_spans'] = [h + t for h, t in zip(df['h_token_spans'], df['t_token_spans'])]
    train_df.rename(columns = {"token": "input", "relation": "label"}, inplace = True)

    model = AutoModelForTokenClassification.from_pretrained(model_name).to(train_device)
    batch_size = 16
    max_num_eigenvectors = model.config.embedding_size // 2 if hasattr(model.config, "embedding_size") else model.config.hidden_size // 2
    os.makedirs("../results", exist_ok = True)

    if test:
        df = df.sample(16)

    ex = df["input"].tolist()
    train_ex = train_df["input"].tolist()
    full_results = []
    full_inputs = tokenizer(train_ex, padding = "max_length", add_special_tokens = False, max_length = model.config.max_position_embeddings - 2 if hasattr(model.config, "max_position_embeddings") else model.config.n_ctx - 2, truncation=True, return_tensors = "pt")

    mlm_model = train_lm_head(
        model_name, 
        train_device, 
        mask_id = tokenizer.mask_token_id, 
        n_epochs = 50, 
        inputs = full_inputs.input_ids, 
        attn_mask = full_inputs.attention_mask,
        encoder_name = encoder_name
    )

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
        running_index: int = 0,
        work_device: torch.device = None,
        mlm_for_pass=None,
        ner_for_pass=None,
        gold_spans: List[List[Tuple[int, int]]] = None,
    ):
        work_device = train_device if work_device is None else _resolve_torch_device(work_device)
        mlm_use = mlm_by_device[work_device] if mlm_for_pass is None else mlm_for_pass
        ner_use = ner_by_device[work_device] if ner_for_pass is None else ner_for_pass
        if getattr(work_device, "type", None) == "cuda":
            torch.cuda.set_device(work_device)

        pullback_dict, entity_span_combinations, entity_spans, valid_document_numbers = forward_backward_pass(
            batch, 
            ner_use, 
            tokenizer, 
            mlm_use, 
            gold_spans=gold_spans, 
            max_num_eigenvectors=max_num_eigenvectors, 
            encoder_name=encoder_name
        )
        if any(res is None for res in [pullback_dict, entity_span_combinations, entity_spans, valid_document_numbers]):
            print("No valid results found for batch", flush = True)
            return [{}]*sum(map(len, batch))

        # Select the correct separators (s_1,s_2,o_1,o_2) for each sentence
        slice_indices = torch.as_tensor([x for x in get_combination_indices_with_repeats(entity_spans)])
        slice_indices = torch.cat([slice_indices, slice_indices+1], dim = -1)[::3, [0,2,1,3]]
        separators = torch.cat([torch.zeros_like(pullback_dict["separators"].squeeze(-1)[...,:1]), pullback_dict["separators"].squeeze(-1)], dim = -1)
        separators = torch.gather(
            separators.cumsum(dim = -1),
            dim = -1,
            index = slice_indices.reshape(-1,4).to(pullback_dict["separators"].device, dtype = torch.int32)
        )

        mi, pointwise_mi = compute_mutual_information(pullback_dict["traces"], pullback_dict["pdets"], pullback_dict["radii"], pullback_dict["cond_probas"], pullback_dict["mask_probas"], separators)
        print(mi.shape)

        entity_eigenvectors = pullback_dict.pop("eigenvectors").detach().cpu()
        radii = pullback_dict.pop("radii").detach().cpu()
        traces = pullback_dict.pop("traces").detach().cpu()

        del separators, slice_indices, pullback_dict, ner_use
        torch.cuda.empty_cache()
        gc.collect()

        # Mean Integrated Directional Gradients baseline
        if with_baselines:
            mean_idg_scores = mean_integrated_directional_gradients(mlm_use, batch, tokenizer, entity_span_combinations, M = 10)
            shap_iq_scores = [tup[0] for tup in shapley_interaction_quantification(mlm_use, batch, tokenizer, entity_spans)]

        current_pos = 0
        batch_results = []
        retokenized_batch = tokenizer(batch, padding = "max_length", add_special_tokens = False, max_length = model.config.max_position_embeddings - 2 if hasattr(model.config, "max_position_embeddings") else model.config.n_ctx - 2, truncation=True, return_tensors = "pt")
        entity_embeddings = mlm_use.model.base_model.embeddings(retokenized_batch.input_ids.to(mlm_use.model.base_model.device)).detach().cpu()

        # Pre-tokenize each sentence once instead of re-tokenizing per (entity, span) pair below.
        batch_token_lists = [tokenizer.tokenize(s, add_special_tokens=False) for s in batch]

        # But eigenvectors, radii and traces are not 1:1 with entity_span_combinations, so we need to re-index them
        for i,sent_span_combinations in enumerate(entity_span_combinations):
            k = len(sent_span_combinations)
            normalized_spans = sorted(set([span for span_combo in sent_span_combinations for span in span_combo]))
            normalized_span_dict = {
                normalized_spans[m]: slice(
                    min(sum([end - start for start, end in normalized_spans[:m]]), normalized_spans[m][0]),
                    sum([end - start for start, end in normalized_spans[m:]]) + normalized_spans[m][1]
                ) 
            for m in range(len(normalized_spans))}

            sent_tokens = batch_token_lists[i]
            entity_pairs_scores = [
                {
                    "e_i_idx": list(span_combo[0]),
                    "e_j_idx": list(span_combo[1]),
                    "e_i_tokens": " ".join(sent_tokens[span_combo[0][0]:span_combo[0][1]]),
                    "e_j_tokens": " ".join(sent_tokens[span_combo[1][0]:span_combo[1][1]]),
                    "score": mi[current_pos + j].to(dtype = torch.float64).item(),
                    "pointwise_baseline": pointwise_mi[current_pos + j].to(dtype = torch.float64).item(),
                    "mean_idg_baseline": mean_idg_scores[current_pos + j].to(dtype = torch.float64).item() if with_baselines else -1.0,
                    "shap_iq_baseline": shap_iq_scores[current_pos + j].to(dtype = torch.float64).item() if with_baselines else -1.0,
                    "original_text": batch[i],
                    "doc_number": running_index + valid_document_numbers[i],
                    "e_i_embeddings": entity_embeddings[i][span_combo[0][0]:span_combo[0][1]],
                    "e_i_eigenvectors": entity_eigenvectors[i][normalized_span_dict[span_combo[0]]],
                    "e_j_embeddings": entity_embeddings[i][span_combo[1][0]:span_combo[1][1]],
                    "e_j_eigenvectors": entity_eigenvectors[i][normalized_span_dict[span_combo[1]]],
                    "e_i_step_size": torch.mean(radii[i][normalized_span_dict[span_combo[0]]] / traces[i][normalized_span_dict[span_combo[0]]]),
                    "e_j_step_size": torch.mean(radii[i][normalized_span_dict[span_combo[1]]] / traces[i][normalized_span_dict[span_combo[1]]])
                }
            for j,span_combo in enumerate(sent_span_combinations)]

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
        work_device = train_device if work_device is None else _resolve_torch_device(work_device)
        if getattr(work_device, "type", None) == "cuda":
            torch.cuda.set_device(work_device)

        # Each entity embedding is a tensor of size (len_entity, embedding_size)
        # Each entity eigenvector is a tensor of size (len_entity, embedding_size, num_eigenvectors)
        # Each entity eigenvalue is a tensor of size (len_entity, num_eigenvectors)

        entity_eigenvectors = [[T[...,-num_pseudorefs:] if num_pseudorefs >= T.shape[-1] else torch.nn.functional.pad(T, (0,num_pseudorefs-T.shape[-1])) for T in L] for L in entity_eigenvectors]

        entity_embeddings = [[emb.unsqueeze(-1).expand(-1, -1, vec.shape[-1]) + vec * step_size for emb, vec, step_size in zip(L1, L2, L3)] for L1,L2,L3 in zip(entity_embeddings, entity_eigenvectors, step_sizes)]

        # HYpotheses are already in the relation disambiguation model
        relation_scores = relation_disambiguation_model(entity_embeddings, batch)

        return [relation_scores[i].as_dict() for i in range(len(relation_scores))]


    def score_batch_list(
        batch_list: List[List[str]],
        running_index: int = 0,
        work_device: torch.device = None,
        mlm_for_pass=None,
        ner_for_pass=None,
        gold_spans: List[List[Tuple[int, int]]] = None,
    ):

        return list(chain(*[score_batch(batch, running_index = running_index + i * batch_size, work_device = work_device, mlm_for_pass = mlm_for_pass, ner_for_pass = ner_for_pass, gold_spans = gold_spans[i*batch_size:(i+1)*batch_size]) for i, batch in enumerate(batch_list)]))

    
    if devices is not None and len(devices) > 1:
        devices = [torch.device(device) for device in devices]
        batches = list(batch_generator(ex, batch_size))
        list_size = (len(batches) + len(devices) - 1) // len(devices)
        batch_lists = [batches[i:i+list_size] for i in range(0, len(batches), list_size)]

        with ThreadPoolExecutor(max_workers = len(devices)) as executor:

            futures = [
                executor.submit(
                    score_batch_list,
                    batch_list,
                    i * batch_size * list_size,
                    devices[i % len(devices)],
                    mlm_for_pass=mlm_by_device[devices[i % len(devices)]],
                    ner_for_pass=ner_by_device[devices[i % len(devices)]],
                    gold_spans=df['gold_spans'].tolist()[i*list_size:(i+1)*list_size]
                )
                for i, batch_list in enumerate(batch_lists)
            ]

            for res in as_completed(futures):
                full_results.extend(res.result())

            print(f"Processed {len(full_results)} examples", flush = True)
            results_to_save = [{k:v for k,v in d.items() if not isinstance(v, torch.Tensor)} for d in full_results]
            with open(f"../results/tacred_scores_{model_name.replace('/', '___')}.json", "w", encoding = "utf-8") as f:
                json.dump(results_to_save, f)
    else:
        for i,batch in enumerate(batch_generator(ex, batch_size)):
            full_results.extend(
                score_batch(
                    batch,
                    running_index=i * batch_size,
                    work_device=train_device,
                    mlm_for_pass=mlm_by_device[train_device],
                    ner_for_pass=ner_by_device[train_device],
                    gold_spans=df['gold_spans'].tolist()[i*batch_size:(i+1)*batch_size]
                )
            )
            print(f"Processed {len(full_results)} examples", flush = True)

            results_to_save = [{k:v for k,v in d.items() if not isinstance(v, torch.Tensor)} for d in full_results]

            with open(f"../results/tacred_scores_{model_name.replace('/', '___')}.json", "w", encoding = "utf-8") as f:
                json.dump(results_to_save, f)

    # MBR DECODING
    full_results_mbrd = []
    if with_mbrd:
        hypotheses = df['label'].unique().tolist() + ["no relation"]
        relation_disambiguation_model = RelationDisambiguation(model_name, hypotheses, topk = 5, batch_size = batch_size)
        grouped_results = list(itertools.groupby(full_results, key = lambda x: x['doc_number']))
        grouped_results = [list(group) for _, group in grouped_results]

        for i,batch in tqdm(enumerate(batch_generator(grouped_results, batch_size)), desc = "MBR Decoding", total = len(grouped_results) + batch_size - 1 // batch_size):
            texts = [L[0]['original_text'] for L in batch]
            entity_embeddings = [[d['entity_embeddings'] for d in L] for L in batch]
            entity_eigenvectors = [[d['entity_eigenvectors'] for d in L] for L in batch]
            step_sizes = [[d['step_size'] for d in L] for L in batch]
            full_results_mbrd.extend(
                mbr_batch(
                    texts,
                    relation_disambiguation_model,
                    entity_embeddings,
                    entity_eigenvectors,
                    step_sizes,
                    num_pseudorefs = 10,
                    work_device = train_device
                )
            )

            print(f"Processed {len(full_results_mbrd)} examples", flush = True)
            with open(f"../results/tacred_scores_{model_name.replace('/', '___')}_mbrd.json", "w", encoding = "utf-8") as f:
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
    main(test=args.test, model_name=args.model_name, with_mbrd=args.mbrd, with_baselines=args.baselines, devices=devices)