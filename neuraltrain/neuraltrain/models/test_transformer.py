# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .transformer import TransformerEncoder, TransformerEncoderConfig
from unittest.mock import MagicMock, patch
from .transformer import LlamaTransformerConfig, LlamaTransformerEncoder


@pytest.fixture
def fake_sequence():
    batch_size = 2
    dim = 64
    n_times = 10
    seq = torch.randn(batch_size, n_times, dim)
    return seq


def test_transformer(fake_sequence):
    batch_size, n_times, dim = fake_sequence.shape

    model = TransformerEncoderConfig().build(dim)
    assert isinstance(model, TransformerEncoder)

    out = model(fake_sequence)
    assert out.shape == (batch_size, n_times, dim)


@pytest.fixture
def mock_llama_layer():
    """Returns a fake decoder layer that passes hidden states through unchanged."""
    layer = MagicMock()
    def _forward(x, **kwargs):
        return (x,)          # LLaMA layers return a tuple; [0] is hidden state
    layer.side_effect = _forward
    return layer

def test_llama_transformer_shape(fake_sequence, mock_llama_layer):
    batch_size, n_times, dim = fake_sequence.shape

    fake_hf_config = MagicMock()
    fake_hf_config.hidden_size = 64   # match dim so projections are square
    fake_hf_config.num_hidden_layers = 2

    fake_model = MagicMock()
    fake_model.model.layers = torch.nn.ModuleList([mock_llama_layer, mock_llama_layer])
    fake_model.model.norm = torch.nn.Identity()

    with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_config), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model):
        cfg = LlamaTransformerConfig(
            model_name="meta-llama/Meta-Llama-3.1-8B",
            num_layers=2,
            torch_dtype="float32",
        )
        model = cfg.build(dim)

    out = model(fake_sequence)
    assert out.shape == (batch_size, n_times, dim)

def test_llama_transformer_padding_mask(mock_llama_layer):
    """Masked positions should not contaminate unmasked output shape."""
    B, T, dim = 2, 5, 64
    x = torch.randn(B, T, dim)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 3:] = True      # last 2 positions are padding

    fake_hf_config = MagicMock()
    fake_hf_config.hidden_size = 64   # match dim so projections are square
    fake_hf_config.num_hidden_layers = 2

    fake_model = MagicMock()
    fake_model.model.layers = torch.nn.ModuleList([mock_llama_layer, mock_llama_layer])
    fake_model.model.norm = torch.nn.Identity()
    with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_config), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model):
        cfg = LlamaTransformerConfig(
            model_name="meta-llama/Meta-Llama-3.1-8B",
            num_layers=2,
            torch_dtype="float32",
        )
        model = cfg.build(dim)
    out = model(x, mask=mask)
    assert out.shape == (B, T, dim)
