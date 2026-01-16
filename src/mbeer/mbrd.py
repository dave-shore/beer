from argparse import OPTIONAL
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import CrossEncoder
import ot.unbalanced as otu
import numpy as np
from mbrs.metrics import Metric
from mbrs.decoders import DecoderMBR


class EmbeddingSearchLayer(torch.nn.Module):
    def __init__(self, vectors: torch.Tensor, p: int = 2):
        super().__init__()
        self.vectors = vectors
        self.p = p

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        return torch.cdist(query, self.vectors, p = self.p).argmin(dim = 1)


class DePositionLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Embedding):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.layer.weight[:x.shape[0]]


class DeTypingLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Embedding):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor, token_type: int) -> torch.Tensor:
        return x - self.layer.weight[:x.shape[0], [token_type]]


class DeNormLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.LayerNorm):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer.elementwise_affine:
            y = (x - self.layer.bias) / (self.layer.weight + 1e-6)
        else:
            y = x

        if getattr(self.layer, "running_mean", None) is not None and getattr(self.layer, "running_var", None) is not None:
            y = y * torch.sqrt(self.layer.running_var + 1e-6) + self.layer.running_mean

        return y


class UnEmbedder(torch.nn.Module):
    def __init__(self, embedding_layer: torch.nn.Module, vocab_size: int, context_size: int, token_types: int = 1, p: int = 2):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.p = p
        self.unembedders = []
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.token_types = token_types

        for module in self.embedding_layer.modules():
            if isinstance(module, torch.nn.Linear):
                new_layers = [
                    torch.nn.Linear(module.out_features, module.in_features, bias = False),
                    torch.nn.Linear(module.out_features, module.out_features, bias = True),
                ]
                new_layers[0].weight = torch.linalg.pinv(module.weight)
                new_layers[1].weight = torch.eye(module.out_features)
                new_layers[1].bias = - module.bias
                self.unembedders.extend(new_layers)

            elif isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == self.vocab_size:
                    self.unembedders.append(EmbeddingSearchLayer(module.weight, p = self.p))
                elif module.weight.shape[0] == self.context_size:
                    self.unembedders.append(DePositionLayer(module))
                else:
                    self.unembedders.append(DeTypingLayer(module))

            elif isinstance(module, torch.nn.LayerNorm):
                self.unembedders.append(DeNormLayer(module))

        self.unembedding = torch.nn.Sequential(*self.unembedders[::-1])

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.unembedding(embeddings)


class CustomMetric(Metric):

  @dataclass
  class Config(Metric.Config):
    model_name_or_path: str = None
    nli_model_name: str = None
    embedding_module_name: str = "embeddings"
    encoder_module_name: str = "encoder"
    classifier_module_name: str = "classifier"
    alignment_type: str = "sinkhorn_stabilized_unbalanced"
    alignment_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

  def __init__(self, cfg):
    super().__init__(cfg)

    self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    self.model = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path)
    self.nli_model = CrossEncoder(cfg.nli_model_name)
    self.embedder = getattr(self.model, cfg.embedding_module_name)
    self.unembedder = UnEmbedder(self.embedder, self.tokenizer.vocab_size, self.model.config.max_position_embeddings)
    self.encoder = getattr(self.model, cfg.encoder_module_name)
    self.classifier = getattr(self.model, cfg.classifier_module_name)
    self.alignment = getattr(otu, cfg.alignment_type)
    self.alignment_kwargs = cfg.alignment_kwargs if cfg.alignment_kwargs else {"reg": 1.0, "reg_m": (0.5, 2.0)}

  def score(
        self, hypothesis: str, references: Tuple[torch.Tensor, torch.Tensor], source: Optional[str] = None
    ) -> float:
        """Calculate the score of the given hypothesis.
        """
        references = [torch.squeeze(ref) for ref in references]
        # shape of each reference tensor is (max_entity_length, embedding_size)

        tokenized_source_and_hp = self.tokenizer([source, hypothesis], padding = True, add_special_tokens = False, return_tensors = "pt")
        with torch.inference_mode():
          embedded_source_and_hp = self.embedder(tokenized_source_and_hp.input_ids)
          embedding_size = embedded_source_and_hp.shape[-1]

          embedded_triple = torch.cat([references[0], embedded_source_and_hp[1][:tokenized_source_and_hp.attention_mask[1].sum()], references[1]], dim = 0)
          mask_embedding = self.embedder(torch.tensor([self.tokenizer.mask_token_id]).to(tokenized_source_and_hp.input_ids.device)).reshape(1, embedding_size)
          embedded_masked_tail = torch.cat([references[0], embedded_source_and_hp[1][:tokenized_source_and_hp.attention_mask[1].sum()], torch.ones_like(references[1]) * mask_embedding], dim = 0)
          embedded_source = embedded_source_and_hp[0]

          padded_input = torch.nn.utils.rnn.pad_sequence([embedded_triple, embedded_source, embedded_masked_tail], batch_first = True)
          output = self.encoder(padded_input).last_hidden_state

        # Now I want to align the first and the second row in a wise manner
        # Alignment 1: source <-> (ref1, hp, ref2)
        # Alignment 2: (ref1, hp, [MASK]) <-> (ref1, hp, ref2)
        # Alignment 3: 
        pair_dists = torch.cdist(
            output[0],
            output[1]
        )
        pair_dists = torch.nn.functional.softmax(pair_dists, dim = 1)
        costs = self.alignment(
            a = torch.where(padded_input[0].sum(dim = 1) > 0, 1, 0).numpy(), 
            b = torch.where(padded_input[1].sum(dim = 1) > 0, 1, 0).numpy(), 
            M = pair_dists, 
            **self.alignment_kwargs
        )
        A = costs.argmax(dim = 1).unsqueeze(1)
        signs = torch.sign(A[-1] - A[0])
        sigma = signs * torch.gather(pair_dists, dim = 1, index = A).mean()

        P_mask = self.classifier(output[2]).softmax(dim = -1)
        P_triple = self.classifier(output[0]).argmax(dim = -1)
        D2 = torch.gather(
            P_mask,
            dim = -1,
            index = P_triple
        )
        gamma = torch.logsumexp(D2)

        unembedded_references = [self.unembedder(ref) for ref in references]
        decoded_references = [self.tokenizer.decode(ref) for ref in unembedded_references]
        non_contradiction = (
            " ".join(decoded_references[0] + [hypothesis] + decoded_references[1]), 
            " ".join(decoded_references[1] + ["NOT " + hypothesis] + decoded_references[0])
        )
        nli_probas = self.nli_model.predict([non_contradiction]).softmax(dim = -1)
        lambda_ = (nli_probas[0][0] - nli_probas[0][2]) * (1 - nli_probas[0][1]) / nli_probas[0][1]
        return gamma / 2 * (lambda_ + sigma)
