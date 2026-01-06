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
from tqdm import tqdm
import gc
import os

hf_api = HfApi()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Handle both relative and absolute imports
try:
    from .pullback import OutputOnlyModel, pullback
    from .utils import timing_decorator
except ImportError:
    # If relative import fails, try absolute import
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from mbeer.pullback import OutputOnlyModel, pullback
    from mbeer.utils import timing_decorator


def pad_traces_and_eigenvectors(L: List[torch.Tensor]) -> torch.Tensor:

    shapes = [t.shape for t in L]
    dims = {len(s) for s in shapes}
    dim = max(dims)
    if len(dims) > 1:
        try:
            force_shape = next(s for s in shapes if len(s) == dim)
            L = [T.reshape((s if s in force_shape else -1 for s in T.shape)) for T in L]
        except:
            raise ValueError("All tensors must have the same number of dimensions or be reshapeable to the same number of dimensions")

    if dim < 3:
        return torch.nn.utils.rnn.pad_sequence([T.unsqueeze(-1) for T in L], batch_first = True, padding_value = 0)

    padding_shape = tuple(max(along_dim) for along_dim in zip(*shapes))

    for i,T in enumerate(L):
        if i % 3:
            # For pairs (j,i), with i < j, we pad the left side of the output dim (dim 1) and the right side of the input dim (dim 2)
            left_padding = tuple(chain.from_iterable([[padding_shape[j]-T.shape[j],0] if j % 2 else [0,padding_shape[j]-T.shape[j]] for j in range(len(T.shape))][::-1]))
            T = torch.nn.functional.pad(T, left_padding).to(dtype = torch.float16)
        else:
            # For pairs (j,i), with i > j, we pad the right side of the output dim (dim 1) and the left side of the input dim (dim 2)
            right_padding = tuple(chain.from_iterable([[0,padding_shape[j]-T.shape[j]] if j % 2 else [padding_shape[j]-T.shape[j],0] for j in range(len(T.shape))][::-1]))
            T = torch.nn.functional.pad(T, right_padding).to(dtype = torch.float16)

        # For tensors that combine two entities it doesn't matter, we treat them as left-padded
        L[i] = T

    return torch.cat(L)
    


@torch.no_grad()
@timing_decorator
def forward_backward_pass(batch: List[str], model: AutoModelForTokenClassification, tokenizer: AutoTokenizer, min_num_eigenvectors: int = 8, max_tokens_per_entity: int = 3, max_retained_vocab: int = 30):

    max_len = model.config.max_position_embeddings - 2 if hasattr(model.config, "max_position_embeddings") else model.config.n_ctx - 2
    pred_to_mask = {k:int(v.startswith(('B','I','L','U'))) for k,v in model.config.id2label.items()} if hasattr(model.config, "id2label") else {}
    device = model.device
    model_name = model.name_or_path

    encoded_inputs = tokenizer(batch, padding = "max_length", add_special_tokens = False, max_length = max_len, truncation=True, return_tensors = "pt")
    attn_mask = encoded_inputs.attention_mask
    row_lens = attn_mask.sum(dim = 1)
    num_labels = len(pred_to_mask) if pred_to_mask else 3
    # Threshold from power of Chi-Squared test for uniformity of distribution
    probit_threshold = 1 / num_labels + sqrt(chi2.sf(0.95, num_labels - 1) / num_labels)

    logit_threshold = torch.nn.Sequential(torch.nn.Softmax(dim = -1), torch.nn.Threshold(probit_threshold, 0))

    predictions = logit_threshold(model(**encoded_inputs.to(device)).logits).argmax(dim = -1).cpu()
    # predictions.shape = (batch_size, max_len)

    # Get the entity spans as (start, end) tuples
    pred_mask = predictions.apply_(pred_to_mask.get) if pred_to_mask else torch.where(predictions > 0, 1, 0)
    ent_spans = [
        torch.diff(seq, prepend = torch.tensor([0])).nonzero(as_tuple = False)
    for seq in pred_mask]
    ent_spans = [
        torch.split(T, 2) if len(T) >= 2 else [torch.zeros(2,1)] for T in ent_spans
    ]
    ent_spans =[
        [(T[0].item(), T[1].item()) if len(T) == 2 else (T[0].item(), row_lens[i].item()) for T in spans]
    for i, spans in enumerate(ent_spans)]
    ent_spans = [
        [T for T in spans if T[0] < T[1]]
    for spans in ent_spans]

    span_combinations = [
        list(combinations(spans, 2))
    for spans in ent_spans]
    # span_combinations.shape = (batch_size, N_combinations(len(spans), 2))

    # Expand the batch so that for each input sentence there are 3*N_combinations(len(spans), 2) inputs: one with the first entity masked, one with the second entity masked, and one with both entities masked
    expanded_batch = torch.cat([
        torch.stack(list(chain(*[
            [
                torch.where(torch.logical_and(torch.arange(encoding.shape[0]) >= s[0], torch.arange(encoding.shape[0]) < s[1]), tokenizer.mask_token_id, encoding),
                torch.where(torch.logical_and(torch.arange(encoding.shape[0]) >= o[0], torch.arange(encoding.shape[0]) < o[1]), tokenizer.mask_token_id, encoding),
                torch.where(torch.logical_or(
                    torch.logical_and(torch.arange(encoding.shape[0]) >= s[0], torch.arange(encoding.shape[0]) < s[1]), 
                    torch.logical_and(torch.arange(encoding.shape[0]) >= o[0], torch.arange(encoding.shape[0]) < o[1])
                ), tokenizer.mask_token_id, encoding)
            ]
        for s,o in span_combo_list])))
        if span_combo_list else torch.empty(0, max_len)
    for encoding, span_combo_list in zip(encoded_inputs.input_ids.to("cpu"), span_combinations)])
    # expanded_batch.shape = (batch_size*3*N_combinations(len(spans), 2), max_len)

    if expanded_batch.shape[0] == 0:
        raise ValueError("No valid span combinations found")

    # Get the ids of the tokens that are perturbed for each input sentence
    eq_class_emb_ids = list(chain(*[
        [
            torch.randint(o[0], o[1], (min(o[1]-o[0], max_tokens_per_entity),)).reshape(-1,1),
            torch.randint(s[0], s[1], (min(s[1]-s[0], max_tokens_per_entity),)).reshape(-1,1),
            torch.cat([torch.randint(s[0], s[1], (min(s[1]-s[0], max_tokens_per_entity),)).reshape(-1,1), torch.randint(o[0], o[1], (min(o[1]-o[0], max_tokens_per_entity),)).reshape(-1,1)]),
        ]
    for span_combo_list in span_combinations for s,o in span_combo_list]))
    eq_class_emb_ids = torch.nn.utils.rnn.pad_sequence(eq_class_emb_ids, batch_first = True, padding_value = -1).squeeze(-1)
    # eq_class_emb_ids.shape = (batch_size*3*N_combinations(len(spans), 2), max_perturbed_entity_spans_length)

    pred_ids = list(chain(*[
        [
            torch.randint(s[0], s[1], (min(s[1]-s[0], max_tokens_per_entity),)).reshape(-1,1),
            torch.randint(o[0], o[1], (min(o[1]-o[0], max_tokens_per_entity),)).reshape(-1,1),
            torch.cat([torch.randint(s[0], s[1], (min(s[1]-s[0], max_tokens_per_entity),)).reshape(-1,1), torch.randint(o[0], o[1], (min(o[1]-o[0], max_tokens_per_entity),)).reshape(-1,1)]),
        ]
    for span_combo_list in span_combinations for s,o in span_combo_list]))
    pred_ids = torch.nn.utils.rnn.pad_sequence(pred_ids, batch_first = True, padding_value = -1).squeeze(-1)
    # pred_ids.shape = (batch_size*3*N_combinations(len(spans), 2), max_perturbed_entity_spans_length)

    separators = list(chain(*[
        [
            s[1] - s[0],
            o[1] - o[0],
            s[1] - s[0]
        ]
    for span_combo_list in span_combinations for s,o in span_combo_list]))

    repetitions = torch.tensor([3*len(span_combo_list) for span_combo_list in span_combinations])
    selected_vocab = torch.cat([
        torch.cat([
            encoding[s[0]:s[1]],
            encoding[o[0]:o[1]]
        ]) for encoding, span_combo_list in zip(encoded_inputs.input_ids.to("cpu"), span_combinations) for s,o in span_combo_list
    ]).unique().reshape(1,1,-1)
    if selected_vocab.shape[-1] > max_retained_vocab:
        selected_vocab = selected_vocab[..., torch.randperm(selected_vocab.shape[-1])[:max_retained_vocab]]
    else:
        selected_vocab = torch.cat([
            selected_vocab,
            torch.randint(0, model.config.vocab_size, (selected_vocab.shape[0], selected_vocab.shape[1], max_retained_vocab - selected_vocab.shape[-1]))
        ], dim = -1)
    selected_vocab = selected_vocab[..., :max_retained_vocab]
    repeated_attn_mask = attn_mask.repeat_interleave(repetitions, dim = 0)

    assert expanded_batch.shape[0] == eq_class_emb_ids.shape[0] == pred_ids.shape[0] == repeated_attn_mask.shape[0]
    local_dataset = TensorDataset(expanded_batch, eq_class_emb_ids, pred_ids, repeated_attn_mask)

    dataloader = DataLoader(local_dataset, batch_size = 2, shuffle = False, pin_memory = True, num_workers = 2)
    parent_model_info = hf_api.model_info(model.name_or_path)
    parent_model_name = parent_model_info.card_data.get("base_model", None)
    del model
    torch.cuda.empty_cache()

    if parent_model_name is not None:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(parent_model_name).to(device, dtype = torch.float16))
    else:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(model_name).to(device, dtype = torch.float16))
        are_weights_initialized ={k:all(torch.all(p.diff() == 0) for n,p in v.named_parameters() if "bias" in n) for k,v in output_only_model.model.named_modules()}
        uninitialized_modules = [k for k,v in are_weights_initialized.items() if v]
        uninitialized_parameters = [p for v in uninitialized_modules for p in output_only_model.model.get_submodule(v).parameters()]

        if len(uninitialized_modules) > 0:
            with torch.enable_grad():
                for module in uninitialized_modules:
                    output_only_model.model.get_submodule(module).train()

                # One step of gradient descent
                loss_fn = torch.nn.CrossEntropyLoss(reduction = "mean")
                optimizer = torch.optim.LBFGS(uninitialized_parameters, lr = 1e-3, history_size = 10, max_iter = 10)
                
                def closure():
                    optimizer.zero_grad()
                    y = output_only_model.model(**encoded_inputs.to(device)).logits
                    loss = loss_fn(
                        torch.nn.functional.one_hot(encoded_inputs.input_ids.to(device), num_classes = output_only_model.model.config.vocab_size).to(y.device, dtype = y.dtype),
                        torch.nn.functional.softmax(y, dim = -1),
                    )
                    loss.backward()
                    return loss
                optimizer.step(closure)

            optimizer.zero_grad(set_to_none = True)
            del optimizer

            for module in uninitialized_modules:
                output_only_model.model.get_submodule(module).eval()

    final_outputs = {
        "traces": [], 
        "eigenvectors": [], 
        "predictions": [],
        "separators": separators
    }

    del encoded_inputs, predictions
    torch.cuda.empty_cache()
    gc.collect()

    for minibatch in tqdm(dataloader):

        minibatch_inputs, minibatch_eq_class_emb_ids, minibatch_pred_ids, minibatch_attn_mask = minibatch
        minibatch_inputs = minibatch_inputs.to(device, dtype = torch.int32)
        minibatch_eq_class_emb_ids = [
            row[row >= 0].unique().tolist()
        for row in minibatch_eq_class_emb_ids]
        minibatch_pred_ids = [
            row[row >= 0].unique().tolist()
        for row in minibatch_pred_ids]
        minibatch_attn_mask = minibatch_attn_mask.to(device, dtype = torch.bool)
        minibatch_selected_vocab = selected_vocab.to(device, dtype = torch.int32).expand(minibatch_inputs.shape[0], minibatch_inputs.shape[1], -1)

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
            min_num_eigenvectors = min_num_eigenvectors
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

        final_outputs["traces"].append(pullback_traces.cpu())
        final_outputs["eigenvectors"].append(diverging_eigenvectors.flatten(start_dim = -2).cpu())
        # Remember to reshape the eigenvectors back to the original shape, since now the last dimension is embedding_size * min_num_eigenvectors
        final_outputs["predictions"].append(predictions.cpu())

        del minibatch_inputs, minibatch_eq_class_emb_ids, minibatch_pred_ids, minibatch_attn_mask, minibatch_selected_vocab, input_embeddings, g
        torch.cuda.empty_cache()
        gc.collect()

    # Outputs must be of shape (len(batch)*N_combinations(len(spans), 2), 3, ...)
    final_outputs = {
        k:torch.stack(
            pad_traces_and_eigenvectors(v).split(3) 
        , dim = 1)
    for k,v in final_outputs.items()}

    return final_outputs

@torch.no_grad()
@timing_decorator
def compute_mutual_information(traces: torch.Tensor, predictions: torch.Tensor, separators: List[int], p_minkowski: float = 2.0):

    # Inputs of shape (len(batch)*N_combinations(len(spans), 2), 3, ...)
    num_per_token_traces = predictions[:,0] * traces[:,1] + predictions[:,1] * traces[:,0]
    den_Pi = torch.nn.utils.rnn.pad_sequence([p[:s] for p,s in zip(predictions, separators)])
    den_Pj = torch.nn.utils.rnn.pad_sequence([p[s:] for p,s in zip(predictions, separators)])
    den_Tij = torch.nn.utils.rnn.pad_sequence([T[:s] for T,s in zip(traces, separators)])
    den_Tji = torch.nn.utils.rnn.pad_sequence([T[s:] for T,s in zip(traces, separators)])
    den_per_token_traces = den_Pi * den_Tij + den_Pj * den_Tji

    agg_num = num_per_token_traces.pow(p_minkowski).prod(dim = -2).sum(dim = -1).pow(1/p_minkowski) / (num_per_token_traces.shape[-1] * num_per_token_traces.shape[-2])
    agg_den = den_per_token_traces.pow(p_minkowski).prod(dim = -2).sum(dim = -1).pow(1/p_minkowski) / (den_per_token_traces.shape[-1] * den_per_token_traces.shape[-2])

    return agg_num / agg_den

@timing_decorator
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_dataset("xiaobendanyn/tacred", split = "test")
    df = pd.DataFrame([json.loads(d['text']) for d in data]).map(lambda x: x['name'] if isinstance(x, dict) else " ".join(x) if isinstance(x, list) else x)
    df.rename(columns = {"token": "input", "relation": "label"}, inplace = True)

    model_name = "rv2307/electra-small-ner"
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ex = df["input"].sample(4).tolist()
    pullback_dict =forward_backward_pass(ex, model, tokenizer)

    mi = compute_mutual_information(pullback_dict["traces"], pullback_dict["predictions"], pullback_dict["separators"])
    print(mi)

if __name__ == "__main__":
    main()