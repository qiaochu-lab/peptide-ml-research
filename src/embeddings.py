"""
ESM-2 peptide embedding module with attention pooling.

Replaces naive mean pooling with a learned attention-weighted pool,
so the model focuses on functionally important residues rather than
treating all positions equally.

Supports:
  - ESM-2-650M  (default, practical on most hardware)
  - ESM-2-3B    (higher capacity, requires >24 GB VRAM)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm


class AttentionPool(nn.Module):
    """
    Learned attention-weighted pooling over token embeddings.

    Instead of averaging all token representations, a small linear
    layer assigns a scalar importance weight to each position.
    This lets the model down-weight padding and up-weight residues
    that carry activity-relevant information.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, L, D)  token embeddings
            mask: (B, L)     True for valid (non-special) tokens
        Returns:
            (B, D)  sequence-level embedding
        """
        scores = self.score(x).squeeze(-1)          # (B, L)
        scores = scores.masked_fill(~mask, -1e9)     # ignore padding / special tokens
        weights = torch.softmax(scores, dim=-1)      # (B, L)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)


class ESMEmbedder:
    """
    Wraps a pre-trained ESM-2 model with attention pooling.

    Usage:
        embedder = ESMEmbedder(model_size="650M")
        vectors  = embedder.encode(["ACDEFG", "KLMNPQ"])  # (2, 1280)
    """

    _LOADERS = {
        "650M": ("esm2_t33_650M_UR50D",  33),
        "3B":   ("esm2_t36_3B_UR50D",    36),
    }

    def __init__(
        self,
        model_size: str = "650M",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        import esm as esm_lib
        loader_name, self.repr_layer = self._LOADERS[model_size]
        loader = getattr(esm_lib.pretrained, loader_name)
        self.model, self.alphabet = loader()
        self.model = self.model.eval().to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = device
        self.batch_size = batch_size
        self.embed_dim = 1280

        # Attention pool is a lightweight add-on trained jointly or fine-tuned
        self.pool = AttentionPool(self.embed_dim).to(device)

    @torch.no_grad()
    def encode(self, sequences: List[str]) -> np.ndarray:
        """
        Encode a list of peptide sequences into fixed-length vectors.

        Returns:
            np.ndarray of shape (N, 1280)
        """
        embeddings = []
        for i in tqdm(range(0, len(sequences), self.batch_size), desc="ESM encoding"):
            batch = sequences[i : i + self.batch_size]
            data = [(f"s{j}", s) for j, s in enumerate(batch)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)

            out = self.model(
                tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False,
            )
            token_emb = out["representations"][self.repr_layer]  # (B, L, D)

            # Mask out <cls>, <eos>, and <pad> tokens before pooling
            pad_idx = self.alphabet.padding_idx
            cls_idx = self.alphabet.cls_idx
            eos_idx = self.alphabet.eos_idx
            mask = (
                (tokens != pad_idx)
                & (tokens != cls_idx)
                & (tokens != eos_idx)
            )  # (B, L)

            pooled = self.pool(token_emb, mask)   # (B, D)
            embeddings.append(pooled.cpu().numpy())

            torch.cuda.empty_cache()

        return np.vstack(embeddings)
