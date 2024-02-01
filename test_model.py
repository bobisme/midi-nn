import torch
import torch.nn as nn

from model import ModelConfig, FeedForward, SelfAttention


def test_feed_forward_params():
    config = ModelConfig(n_embeds=64, dropout=0.13)
    ff = FeedForward(config)
    layers = list(ff.net.modules())
    assert isinstance(layers[1], nn.Linear)
    assert layers[1].weight.shape == (256, 64)
    assert isinstance(layers[2], nn.GELU)
    assert isinstance(layers[3], nn.Linear)
    assert layers[3].weight.shape == (64, 256)
    assert isinstance(layers[4], nn.Dropout)
    assert layers[4].p == 0.13


def test_self_attention_params():
    config = ModelConfig(context_len=8, n_embeds=64, n_heads=4, dropout=0.13)
    sa = SelfAttention(config)
    assert sa.flash
    assert sa.key.weight.shape == (16, 64)
    assert sa.query.weight.shape == (16, 64)
    assert sa.value.weight.shape == (16, 64)
    assert sa.dropout.p == 0.13
    mask = sa.get_buffer("mask")
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )
