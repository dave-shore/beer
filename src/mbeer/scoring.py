from types import NoneType
from typing import List
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations, chain
import pandas as pd
import json
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

hf_api = HfApi()

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

@timing_decorator
def forward_backward_pass(batch: List[str], model: AutoModelForTokenClassification, tokenizer: AutoTokenizer):

    max_len = model.config.max_position_embeddings - 2 if hasattr(model.config, "max_position_embeddings") else model.config.n_ctx - 2
    pred_to_mask = {k:int(v.startswith(('B','I','L','U'))) for k,v in model.config.id2label.items()}

    encoded_inputs = tokenizer(batch, padding = "max_length", add_special_tokens = False, max_length = max_len, truncation=True, return_tensors = "pt")
    attn_mask = encoded_inputs.attention_mask
    row_lens = attn_mask.sum(dim = 1)

    predictions = model(**encoded_inputs).logits.argmax(dim = -1)
    # predictions.shape = (batch_size, max_len)

    # Get the entity spans as (start, end) tuples
    pred_mask = predictions.apply_(pred_to_mask.get)
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
    for encoding, span_combo_list in zip(encoded_inputs.input_ids, span_combinations)])
    # expanded_batch.shape = (batch_size*3*N_combinations(len(spans), 2), max_len)

    if expanded_batch.shape[0] == 0:
        raise ValueError("No valid span combinations found")

    # Get the ids of the tokens that are perturbed for each input sentence
    eq_class_emb_ids = list(chain(*[
        [
            torch.arange(o[0], o[1]).reshape(-1,1),
            torch.arange(s[0], s[1]).reshape(-1,1),
            torch.cat([torch.arange(s[0], s[1]).reshape(-1,1), torch.arange(o[0],o[1]).reshape(-1,1)]),
        ]
    for span_combo_list in span_combinations for s,o in span_combo_list]))
    eq_class_emb_ids = torch.nn.utils.rnn.pad_sequence(eq_class_emb_ids, batch_first = True, padding_value = -1).squeeze(-1)
    # eq_class_emb_ids.shape = (batch_size*3*N_combinations(len(spans), 2), max_perturbed_entity_spans_length)

    pred_ids = list(chain(*[
        [
            torch.arange(s[0], s[1]).reshape(-1,1),
            torch.arange(o[0], o[1]).reshape(-1,1),
            torch.cat([torch.arange(s[0], s[1]).reshape(-1,1), torch.arange(o[0],o[1]).reshape(-1,1)]),
        ]
    for span_combo_list in span_combinations for s,o in span_combo_list]))
    pred_ids = torch.nn.utils.rnn.pad_sequence(pred_ids, batch_first = True, padding_value = -1).squeeze(-1)
    # pred_ids.shape = (batch_size*3*N_combinations(len(spans), 2), max_perturbed_entity_spans_length)

    repetitions = torch.tensor([3*len(span_combo_list) for span_combo_list in span_combinations])
    selected_vocab = torch.cat([
        torch.cat([
            encoding[s[0]:s[1]],
            encoding[o[0]:o[1]]
        ]) for encoding, span_combo_list in zip(encoded_inputs.input_ids, span_combinations) for s,o in span_combo_list
    ]).unique().reshape(1,1,-1)

    repeated_attn_mask = attn_mask.repeat_interleave(repetitions, dim = 0)

    assert expanded_batch.shape[0] == eq_class_emb_ids.shape[0] == pred_ids.shape[0] == repeated_attn_mask.shape[0]
    local_dataset = TensorDataset(expanded_batch, eq_class_emb_ids, pred_ids, repeated_attn_mask)

    dataloader = DataLoader(local_dataset, batch_size = 2, shuffle = False, pin_memory = True)
    parent_model_info = hf_api.model_info(model.name_or_path)
    parent_model_name = parent_model_info.card_data.get("base_model", None)

    if parent_model_name is None:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(model.name_or_path).to(model.device))
    else:
        output_only_model = OutputOnlyModel(AutoModelForMaskedLM.from_pretrained(parent_model_name).to(model.device))

    for minibatch in tqdm(dataloader):
        minibatch_inputs, minibatch_eq_class_emb_ids, minibatch_pred_ids, minibatch_attn_mask = minibatch
        minibatch_inputs = minibatch_inputs.to(model.device, dtype = torch.long)
        minibatch_eq_class_emb_ids = [
            [v.item() for v in row if v >= 0]
        for row in minibatch_eq_class_emb_ids]
        minibatch_pred_ids = [
            [v.item() for v in row if v >= 0]
        for row in minibatch_pred_ids]
        minibatch_attn_mask = minibatch_attn_mask.to(model.device, dtype = torch.bool)
        selected_vocab = selected_vocab.tile(minibatch_inputs.shape[0], minibatch_inputs.shape[1], 1)

        if hasattr(model.base_model, "embeddings"):
            input_embeddings = model.base_model.embeddings(minibatch_inputs)
        else:
            input_embeddings = model.base_model.embed_tokens(minibatch_inputs)
        # input_embeddings.shape = (batch_size*3*N_combinations(len(spans), 2), max_len, embedding_size)

        pullback_traces, diverging_eigenvectors = pullback(
            input_embeddings,
            g = torch.eye(model.config.hidden_size, device = input_embeddings.device),
            model = output_only_model,
            eq_class_emb_ids = minibatch_eq_class_emb_ids,
            pred_id = minibatch_pred_ids,
            select = selected_vocab,
            attention_mask = minibatch_attn_mask,
            return_trace = True
        )
        tqdm.write(f"Pullback traces shape: {pullback_traces.shape}")
        tqdm.write(f"Diverging eigenvectors shape: {diverging_eigenvectors.shape}")


@timing_decorator
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_dataset("xiaobendanyn/tacred", split = "test")
    df = pd.DataFrame([json.loads(d['text']) for d in data]).map(lambda x: x['name'] if isinstance(x, dict) else " ".join(x) if isinstance(x, list) else x)
    df.rename(columns = {"token": "input", "relation": "label"}, inplace = True)

    model_name = "Babelscape/wikineural-multilingual-ner"
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ex = df["input"].sample(4).tolist()
    forward_backward_pass(ex, model, tokenizer)

if __name__ == "__main__":
    main()