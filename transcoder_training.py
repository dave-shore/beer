from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
import huggingface_hub as hfh
from datasets import load_dataset, Dataset
import torch
import pandas as pd
from circuit_tracer import ReplacementModel
from transformer_lens import HookedTransformerConfig
import lightning as L
from sklearn.model_selection import train_test_split


class LightingWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module, ref_model: torch.nn.Module, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.ref_model = ref_model.to("cpu")
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer

    def compute_loss(self, inputs):
        self.ref_model = self.ref_model.to("cpu")
        try:
            outputs = self.model(inputs['input'])
            reembedded_inputs = self.model.embed(outputs.argmax(dim = -1))
            targets = self.ref_model(**self.tokenizer(inputs['input'], padding = True, return_tensors = "pt"), output_hidden_states = True).hidden_states[-1].to(reembedded_inputs.device)
        except Exception as e:
            print(inputs)
            raise e
        min_seq_len = min(reembedded_inputs.shape[1], targets.shape[1])
        return torch.nn.functional.mse_loss(reembedded_inputs[:, :min_seq_len], targets[:, :min_seq_len], reduction = "mean")

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


def get_model_training_data(model_name):
    dataset_names = hfh.model_info(model_name).card_data['datasets']
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    for dataset_name in dataset_names:
        data_train = pd.concat([data_train, load_dataset(dataset_name).to_pandas()])
    
    if "tokens" in data_train.columns:
        data_train.rename(columns = {"tokens": "input"}, inplace = True)
    elif "token" in data_train.columns:
        data_train.rename(columns = {"token": "input"}, inplace = True)
    elif "text" in data_train.columns:
        data_train.rename(columns = {"text": "input"}, inplace = True)
    else:
        raise ValueError(f"No valid input column found in {dataset_name}")

    data_train['input'] = data_train['input'].apply(lambda x: x['name'] if isinstance(x, dict) else " ".join(x) if isinstance(x, list) else x)
    
    data_train['label_ids'] = [[i] for i in range(len(data_train))]
    return data_train
        

def main():

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = "Babelscape/wikineural-multilingual-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    df = get_model_training_data(model_name)
    train_df, eval_df = train_test_split(df, test_size = 0.1, random_state = 42)
    train_dataset = Dataset.from_pandas(train_df[['input', 'label_ids']])
    eval_dataset = Dataset.from_pandas(eval_df[['input', 'label_ids']])

    config = {k:v for k,v in model.config.__dict__.items() if k in HookedTransformerConfig.__dataclass_fields__.keys()}
    config["n_ctx"] = model.config.max_position_embeddings
    config["n_layers"] = model.config.num_hidden_layers
    config["d_model"] = model.config.hidden_size
    config["d_head"] = model.config.hidden_size // model.config.num_attention_heads
    config["act_fn"] = "gelu"

    transducer = ReplacementModel(config, tokenizer, default_padding_side = "right")
    transducer = transducer.to(device)
    LWrapper = LightingWrapper(transducer, model, tokenizer)

    trainer = L.Trainer(
        precision = "bf16",
        gradient_clip_val = 1.0,
        max_epochs = 100,
        model_registry = "Transducer",
        devices = "auto"
    )
    trainer.fit(LWrapper, train_dataset, eval_dataset)

    try:
        torch.save(transducer, "transducer.pt")
    except Exception as e:
        print(e)
        print("Failed to save transducer")
        print("Saving transducer state_dict only")
        torch.save(transducer.state_dict(), "transducer.pth")

