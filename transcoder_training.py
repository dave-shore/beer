from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel
import huggingface_hub as hfh
from datasets import load_dataset, Dataset, DatasetDict
import torch
import pandas as pd
from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder import TranscoderSet, SingleLayerTranscoder
from circuit_tracer.transcoder.activation_functions import JumpReLU
from transformer_lens import HookedTransformerConfig
import lightning as L
from sklearn.model_selection import train_test_split
import logging
import argparse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcoder_training_logs.log'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)


class LightingWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module, ref_model: torch.nn.Module, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            if param.is_leaf:
                param.requires_grad = True
        self.model.hook_embed.remove_hooks(dir = "both", including_permanent=True)

        self.ref_model = ref_model.to("cpu")
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer
        self.max_len = self.ref_model.config.max_position_embeddings - 2 if hasattr(self.ref_model.config, "max_position_embeddings") else self.ref_model.config.n_ctx - 2
        
    def compute_loss(self, inputs):
        try:
            encoded_inputs = self.tokenizer(inputs['input'], padding = "max_length", add_special_tokens = False, max_length = self.max_len, truncation=True, return_tensors = "pt").input_ids
            outputs = self.model(encoded_inputs)
            reembedded_inputs = self.model.embed(outputs.argmax(dim = -1))
            targets = self.ref_model(**encoded_inputs, output_hidden_states = True).hidden_states[-1].to(reembedded_inputs.device)
        except ValueError as e:
            logger.error(f"Error in compute_loss: {e}")
            logger.debug(f"Inputs that caused error: {inputs}")
            raise e
        min_seq_len = min(reembedded_inputs.shape[1], targets.shape[1])
        loss = torch.nn.functional.mse_loss(reembedded_inputs[:, :min_seq_len], targets[:, :min_seq_len], reduction = "mean")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        if batch_idx % 100 == 0:  # Log every 100 batches to avoid too much logging
            logger.info(f"Training batch {batch_idx}, loss: {loss.item():.4f}")
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


def get_model_training_data(model_name):
    logger.info(f"Loading training data for model: {model_name}")
    dataset_names = hfh.model_info(model_name).card_data['datasets']
    logger.info(f"Found {len(dataset_names)} dataset(s): {dataset_names}")
    data_train = pd.DataFrame()
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        if isinstance(dataset_name, str):
            dataset_data = load_dataset(dataset_name)
            if isinstance(dataset_data, Dataset):
                dataset_data = dataset_data.to_pandas()
            elif isinstance(dataset_data, DatasetDict):
                dataset_data = pd.concat([v.to_pandas() for v in dataset_data.values()])
            else:
                raise ValueError(f"Unknown dataset type: {type(dataset_data)}")
        else:
            continue
        logger.info(f"Loaded {len(dataset_data)} samples from {dataset_name}")
        data_train = pd.concat([data_train, dataset_data])
    
    logger.info(f"Total training samples: {len(data_train)}")
    
    if "tokens" in data_train.columns:
        data_train.rename(columns = {"tokens": "input"}, inplace = True)
        logger.debug("Renamed 'tokens' column to 'input'")
    elif "token" in data_train.columns:
        data_train.rename(columns = {"token": "input"}, inplace = True)
        logger.debug("Renamed 'token' column to 'input'")
    elif "text" in data_train.columns:
        data_train.rename(columns = {"text": "input"}, inplace = True)
        logger.debug("Renamed 'text' column to 'input'")
    else:
        error_msg = f"No valid input column found in {dataset_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Processing input column data")
    data_train['input'] = data_train['input'].apply(lambda x: x['name'] if isinstance(x, dict) else " ".join(x) if isinstance(x, list) else x)
    
    data_train['label_ids'] = [[i] for i in range(len(data_train))]
    logger.info(f"Data preparation complete. Final dataset size: {len(data_train)}")
    return data_train
        

def main(model_name: str = "Babelscape/wikineural-multilingual-ner", single_gpu: bool = False):
    logger.info("="*80)
    logger.info("Starting transcoder training")
    logger.info("="*80)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Set NCCL environment variables to help with CUDA/NCCL compatibility issues
    # These can help resolve "named symbol not found" errors
    os.environ.setdefault("NCCL_DEBUG", "WARN")  # Set to INFO for more details
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # Disable P2P if there are GPU communication issues
    os.environ.setdefault("NCCL_IB_DISABLE", "1")  # Disable InfiniBand if not available
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")  # Use loopback interface if needed
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}, with PyTorch version: {torch.__version__}, and CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    logger.info(f"Model loaded. Model config: {model.config}")
    
    df = get_model_training_data(model_name)
    logger.info("Splitting dataset into train and evaluation sets")
    train_df, eval_df = train_test_split(df, test_size = 0.1, random_state = 42)
    logger.info(f"Train samples: {len(train_df)}, Eval samples: {len(eval_df)}")
    train_dataset = Dataset.from_pandas(train_df[['input', 'label_ids']])
    eval_dataset = Dataset.from_pandas(eval_df[['input', 'label_ids']])

    logger.info("Creating transducer configuration")
    config = {k:v for k,v in model.config.__dict__.items() if k in HookedTransformerConfig.__dataclass_fields__.keys()}
    config["n_ctx"] = model.config.max_position_embeddings
    config["n_layers"] = model.config.num_hidden_layers
    config["d_model"] = model.config.hidden_size
    config["d_head"] = model.config.hidden_size // model.config.num_attention_heads
    config["act_fn"] = "gelu"
    logger.info(f"Transducer config: {config}")

    logger.info("Initializing ReplacementModel (transducer)")
    transcoder_set = TranscoderSet(
        {layer_idx: SingleLayerTranscoder(config["d_model"], config["d_model"] * 2, JumpReLU(torch.tensor(0.0), 0.1), layer_idx) for layer_idx in range(config["n_layers"])},
        feature_input_hook = "hook_resid_pre",
        feature_output_hook = "hook_mlp_out"
    )
    transducer = ReplacementModel.from_config(config, transcoders = transcoder_set, tokenizer=tokenizer)    
    logger.info("Creating LightningWrapper")
    LWrapper = LightingWrapper(transducer, model, tokenizer)

    logger.info("Initializing PyTorch Lightning Trainer")    
    
    if single_gpu or num_gpus <= 1:
        logger.info("Using single device training")
        devices = 1
        strategy = "auto"
    else:
        # For multi-GPU, try DDP but with fallback options
        logger.info(f"Detected {num_gpus} GPUs. Attempting multi-GPU training with DDP.")
        devices = "auto"
        strategy = "ddp_find_unused_parameters_true"  # Use DDP for multi-GPU
    
    trainer = L.Trainer(
        precision = "bf16-mixed",
        gradient_clip_val = 1.0,
        max_epochs = 100,
        min_epochs = 1,
        model_registry = "Transducer",
        devices = devices,
        strategy = strategy
    )
    logger.info(f"Trainer configuration: max_epochs={trainer.max_epochs}, precision={trainer.precision}, gradient_clip_val={trainer.gradient_clip_val}, strategy={strategy}, devices={devices}")
    
    logger.info("Starting training...")
    trainer.fit(LWrapper, train_dataset, eval_dataset)
    logger.info("Training completed successfully")

    logger.info("Saving transducer model")
    try:
        torch.save(transducer, "transducer.pt")
        logger.info("Transducer saved successfully to transducer.pt")
    except Exception as e:
        logger.error(f"Failed to save transducer: {e}")
        logger.warning("Attempting to save transducer state_dict only")
        try:
            torch.save(transducer.state_dict(), "transducer.pth")
            logger.info("Transducer state_dict saved successfully to transducer.pth")
        except Exception as e2:
            logger.error(f"Failed to save transducer state_dict: {e2}")
    
    logger.info("="*80)
    logger.info("Transcoder training finished")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transcoder model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Babelscape/wikineural-multilingual-ner",
        help="Name of the model to use for training (default: Babelscape/wikineural-multilingual-ner)"
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Force single GPU training (useful if NCCL/distributed training fails)"
    )
    args = parser.parse_args()
    main(model_name=args.model_name, single_gpu=args.single_gpu)
