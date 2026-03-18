# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer models
"""

import logging
import typing as tp

import torch
from torch import nn
from x_transformers import Encoder  # type: ignore

from .base import BaseModelConfig

logger = logging.getLogger(__name__)


class TransformerEncoderConfig(BaseModelConfig):
    name: tp.Literal["TransformerEncoder"] = "TransformerEncoder"
    heads: int = 8
    depth: int = 12
    attn_dropout: float = 0.1
    ff_dropout: float = 0.0
    use_scalenorm: bool = True
    rotary_pos_emb: bool = True
    use_rmsnorm: bool = False
    residual_attn: bool = False
    scale_residual: bool = True
    resi_dual: bool = True

    def build(self, dim: int) -> nn.Module:
        return TransformerEncoder(dim, config=self)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder model based on x-transformers:
    https://github.com/lucidrains/x-transformers

    """

    def __init__(
        self,
        # Channels
        dim: int,
        config: TransformerEncoderConfig | None = None,
    ):
        super().__init__()
        config = config if config is not None else TransformerEncoderConfig()

        if dim % config.heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by the number of heads ({config.heads})"
            )

        self.transformer_encoder = Encoder(
            dim=dim,
            depth=config.depth,
            heads=config.heads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            use_scalenorm=config.use_scalenorm,
            use_rmsnorm=config.use_rmsnorm,
            rotary_pos_emb=config.rotary_pos_emb,
            residual_attn=config.residual_attn,
            scale_residual=config.scale_residual,
            resi_dual=config.resi_dual,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.transformer_encoder.forward(
            x,
            mask=mask,
        )

        return x

# ---------------------------------------------------------------------------
# LLaMA 3.1 8B-based encoder
# ---------------------------------------------------------------------------
 
class LlamaTransformerConfig(BaseModelConfig):
    """Configuration for a transformer encoder backed by LLaMA 3.1 8B.
 
    The LLaMA token-embedding table and language-model head are discarded.
    Only the transformer decoder layers and the final RMSNorm are retained.
    Two linear projections bridge the brain-signal embedding dimension and
    LLaMA's internal hidden dimension (4 096 for the 8B model).
 
    Parameters
    ----------
    model_name :
        HuggingFace model ID or local path.  Must point to a LLaMA 3.1 8B
        checkpoint for which access has already been granted.
    num_layers :
        How many of LLaMA's 32 decoder layers to keep.  ``None`` keeps all
        32.  Set to e.g. 16 to match the layer count used in the original
        pipeline.
    freeze_pretrained :
        If ``True`` (default), the loaded LLaMA weights are frozen and only
        the input/output projections are trained.  Set to ``False`` to
        fine-tune the entire stack.
    torch_dtype :
        Precision used when loading the pretrained weights.  ``"bfloat16"``
        halves memory relative to ``"float32"`` and is the recommended
        choice for A100/H100 GPUs.
    """
 
    name: tp.Literal["LlamaTransformer"] = "LlamaTransformer"
    model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    num_layers: int | None = None       # None → use all 32 layers
    freeze_pretrained: bool = True
    torch_dtype: str = "bfloat16"       # "float32" | "float16" | "bfloat16"
 
    def build(self, dim: int) -> nn.Module:
        return LlamaTransformerEncoder(dim, config=self)
 
 
class LlamaTransformerEncoder(nn.Module):
    """Bidirectional encoder built from LLaMA 3.1 8B transformer blocks.
 
    Architecture overview
    ---------------------
    brain features  (B, T, dim)
         │
         │  input_proj  [trainable]   dim → llama_hidden_size
         ▼
    LLaMA decoder layers 0 … N-1     full (non-causal) self-attention
         │  + final RMSNorm
         │  output_proj [trainable]   llama_hidden_size → dim
         ▼
    brain features  (B, T, dim)
 
    Causal masking is disabled: every position can attend to every other
    position, which is appropriate for the sentence-level encoder role
    described in the paper (the model has access to the full sentence
    window at inference time).
 
    Parameters
    ----------
    dim :
        Input / output feature dimension (= word-embedding dimension, 1024).
    config :
        ``LlamaTransformerConfig`` instance controlling model selection,
        layer count, dtype and freezing behaviour.
    """
 
    def __init__(
        self,
        dim: int,
        config: LlamaTransformerConfig | None = None,
    ):
        super().__init__()
        config = config if config is not None else LlamaTransformerConfig()
 
        # ------------------------------------------------------------------ #
        # Load pretrained LLaMA weights
        # ------------------------------------------------------------------ #
        from transformers import AutoConfig, AutoModelForCausalLM  # lazy import
 
        _DTYPE_MAP: dict[str, torch.dtype] = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }
        load_dtype = _DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)
 
        logger.info(
            "LlamaTransformerEncoder: loading '%s' (dtype=%s) …",
            config.model_name,
            config.torch_dtype,
        )
 
        llama_hf_config = AutoConfig.from_pretrained(config.model_name)
        llama_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=load_dtype,
        )
 
        # ------------------------------------------------------------------ #
        # Extract the pieces we need and discard the rest
        # ------------------------------------------------------------------ #
        self.llama_hidden_size: int = llama_hf_config.hidden_size  # 4 096 for 8B
 
        n_total_layers: int = llama_hf_config.num_hidden_layers       # 32 for 8B
        n_keep: int = (
            config.num_layers if config.num_layers is not None else n_total_layers
        )
        if n_keep > n_total_layers:
            raise ValueError(
                f"num_layers={n_keep} exceeds the model's {n_total_layers} layers."
            )
 
        # Decoder blocks and the shared final RMSNorm
        self.layers = nn.ModuleList(llama_model.model.layers[:n_keep])
        self.norm   = llama_model.model.norm
 
        # Free the embedding table, LM head and any remaining decoder blocks
        del llama_model
 
        logger.info(
            "LlamaTransformerEncoder: retaining %d / %d decoder layers.",
            n_keep,
            n_total_layers,
        )
 
        # ------------------------------------------------------------------ #
        # Optionally freeze the pretrained weights
        # ------------------------------------------------------------------ #
        if config.freeze_pretrained:
            for param in self.layers.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False
            logger.info("LlamaTransformerEncoder: pretrained weights frozen.")
 
        # ------------------------------------------------------------------ #
        # Learnable dimension-adapting projections (always trained)
        # ------------------------------------------------------------------ #
        # These run in float32 (the outer pipeline's default precision) and
        # are placed *outside* the bfloat16 region so gradients stay clean.
        self.input_proj  = nn.Linear(dim, self.llama_hidden_size)
        self.output_proj = nn.Linear(self.llama_hidden_size, dim)
 
    # ---------------------------------------------------------------------- #
 
    @staticmethod
    def _build_bidirectional_mask(
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a 4-D additive attention mask with no causal restrictions.
 
        LLaMA decoder layers expect an additive mask of shape
        ``(B, 1, T_q, T_k)`` where ``0.0`` means "attend" and a large
        negative value means "do not attend".  Passing an all-zeros tensor
        gives full bidirectional (non-causal) attention.
 
        Parameters
        ----------
        padding_mask :
            Optional boolean tensor ``(B, T)`` where ``True`` marks *valid*
            (non-padding) positions.  Padding positions are set to ``-inf``
            so they are never attended to.
        """
        mask = torch.zeros(
            batch_size, 1, seq_len, seq_len,
            dtype=dtype, device=device,
        )
        if padding_mask is not None:
            # padding_mask: (B, T), True = valid → False = pad
            neg_inf = torch.finfo(dtype).min
            pad = (~padding_mask).to(dtype) * neg_inf  # (B, T)
            # Broadcast over the query dimension: padded *keys* are masked out
            mask = mask + pad[:, None, None, :]         # (B, 1, 1, T)
        return mask
 
    # ---------------------------------------------------------------------- #
 
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x :
            Brain feature tensor of shape ``(batch_size, seq_len, dim)``.
        mask :
            Optional boolean padding mask ``(batch_size, seq_len)`` where
            ``True`` marks valid (non-padding) positions.
 
        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, seq_len, dim)``.
        """
        B, T, _ = x.shape
 
        # ---- project to LLaMA hidden dimension (float32 → load_dtype) ---- #
        x = self.input_proj(x)                            # (B, T, llama_hidden_size)
        llama_dtype = next(self.layers.parameters()).dtype
        x = x.to(llama_dtype)
 
        # ---- position ids ------------------------------------------------- #
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
 
        # ---- bidirectional attention mask --------------------------------- #
        attn_mask_4d = self._build_bidirectional_mask(
            B, T, dtype=llama_dtype, device=x.device, padding_mask=mask,
        )
 
        # ---- cache_position (required by transformers >= 4.40) ----------- #
        cache_position = torch.arange(T, device=x.device)
 
        # ---- run through LLaMA decoder layers ----------------------------- #
        for layer in self.layers:
            # Support both older (<4.40) and newer (>=4.40) transformers APIs
            try:
                layer_out = layer(
                    x,
                    attention_mask=attn_mask_4d,
                    position_ids=position_ids,
                    use_cache=False,
                    cache_position=cache_position,
                )
            except TypeError:
                # Older transformers: no cache_position argument
                layer_out = layer(
                    x,
                    attention_mask=attn_mask_4d,
                    position_ids=position_ids,
                    use_cache=False,
                )
            x = layer_out[0]
 
        # ---- final LLaMA RMSNorm ------------------------------------------ #
        x = self.norm(x)
 
        # ---- project back to brain-signal embedding dim (float32) --------- #
        x = x.to(torch.float32)
        x = self.output_proj(x)                           # (B, T, dim)
 
        return x