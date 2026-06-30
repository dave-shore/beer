import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PCAHeadOutput:
    logits: torch.Tensor
    pve_register: List[float]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.logits.shape

    @property
    def device(self) -> torch.device:
        return self.logits.device

    @property
    def dtype(self) -> torch.dtype:
        return self.logits.dtype


class PCAHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.reduced_dim = out_features
        self.pve_register = []
        self.device = device
        self.dtype = dtype
        self.linear = torch.empty(in_features, out_features, device=device, dtype=dtype)

    def to(self, device: torch.device, dtype: torch.dtype | None = None):
        self.device = device
        self.dtype = dtype if dtype is not None else self.dtype
        return self

    def forward(self, X: torch.Tensor) -> PCAHeadOutput:

        X_flat = X.reshape(-1, X.shape[-1]).nan_to_num(0.0)

        with torch.no_grad():
            _, S, Vh = torch.linalg.svd(X_flat, full_matrices=True)
            self.pve_register.append(S[:self.reduced_dim].pow(2).sum() / S.pow(2).sum())
            Vh = Vh.contiguous().T

        if Vh.shape[1] > self.reduced_dim:
            Vh = Vh[:, :self.reduced_dim]
        else:
            Vh = torch.nn.functional.pad(Vh, (0, self.reduced_dim - Vh.shape[1]))

        self.linear = Vh.clone()
        logits = X @ self.linear

        return PCAHeadOutput(logits=logits, pve_register=self.pve_register)


class PCAHeadModel(nn.Module):
    """Transformer wrapper that applies PCAHead as the final forward step."""

    def __init__(self, base_model: nn.Module, pca_head: PCAHead):
        super().__init__()
        self.model = base_model
        self.base_model = base_model.base_model if hasattr(base_model, "base_model") else base_model
        self.lm_head = pca_head.to(base_model.device, base_model.dtype)
        self.config = base_model.config
        self.device = base_model.device
        self.dtype = base_model.dtype

    def to(self, device: torch.device, dtype: torch.dtype | None = None):
        self.device = device
        self.dtype = dtype if dtype is not None else self.dtype
        self.model = self.model.to(self.device, self.dtype)
        self.lm_head = self.lm_head.to(self.device, self.dtype)
        return self

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        return self.lm_head(hidden_states)
