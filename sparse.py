import torch
from torch.functional import Tensor


def reshape_mask(mask: Tensor):
    n = mask.size(0)
    return torch.reshape(mask, [1, 1, n, n])


def get_full_attention_mask(size: int):
    return torch.tril(torch.ones([size, size]))


def get_mask_with_bandwidth(size: int, bandwidth: int):
    ctx = min(size - 1, bandwidth - 1)
    mask = torch.tril(torch.ones([size, size]), ctx)
    return mask


def get_mask_band(size: int, bandwidth: int):
    return get_mask_with_bandwidth(size, 1) - get_mask_with_bandwidth(
        size, 1 - bandwidth
    )


def get_strided_mask(size: int, stride: int):
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cols = torch.arange(size)
    rows = cols[:, None]
    mask = (rows >= cols) & ((rows - cols) % stride == 0)
    return mask.float()


def strided_transpose(x, local_attn_ctx):
    # assert bT_ctx % blocksize == 0, f"{bT_ctx}, {blocksize}"
    # bT_ctx = n_ctx // local_attn_ctx
    n, t, embd = x.size()
    bT_ctx = t // local_attn_ctx
    x = x.view(n, bT_ctx, local_attn_ctx, embd)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(n, t, embd)
    return x


def fixed_attention_mask(seq_len: int, stride: int) -> torch.Tensor:
    verticals = torch.arange(1 - seq_len, 1) % stride == 0
    cols = torch.arange(-seq_len, 0)
    blocks = (cols // stride - cols[:, None] // stride) == 0
    mask = torch.tril(verticals | blocks)
    return mask.float()
