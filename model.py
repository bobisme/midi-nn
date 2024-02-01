from collections import namedtuple
import math
from dataclasses import dataclass

from rich.progress import track
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

END_SONG_TOKEN = 3


@dataclass
class ModelConfig:
    context_len: int = 512
    n_layers: int = 128
    n_heads: int = 6
    n_embeds: int = 384
    n_tokens: int = 519
    n_added_params: int = 2
    n_track_pos_embeds: int = 512
    dropout: float = 0.0
    bias: bool = False

    @property
    def n_embed_and_added(self):
        return self.n_embeds + self.n_added_params

    @property
    def n_tokens_and_added(self):
        return self.n_tokens + self.n_added_params


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, flash=True):
        super().__init__()
        assert config.n_embeds % config.n_heads == 0
        self.config = config
        self.flash = flash
        head_size = config.n_embeds // config.n_heads
        self.key = nn.Linear(config.n_embeds, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embeds, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embeds, head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.context_len, config.context_len))
        )

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        if self.flash:
            dropout = self.config.dropout if self.training else 0.0
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True
            )
        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        head_size = config.n_embeds // config.n_heads
        self.heads = nn.ModuleList(
            [SelfAttention(config) for _ in range(config.n_heads)]
        )
        self.projection = nn.Linear(
            config.n_heads * head_size, config.n_embeds, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class MultiHeadAttentionV2(nn.Module):
    def __init__(self, config: ModelConfig, flash=True):
        super().__init__()
        assert config.n_embeds % config.n_heads == 0
        self.config = config
        self.flash = flash
        self.key = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.query = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.value = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.context_len, config.context_len))
        )

    def _attend(
        self, shape: torch.Size, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = shape
        if self.flash:
            dropout = self.config.dropout if self.training else 0.0
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True
            )
        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ v

    def forward(self, x):
        B, T, C = x.shape

        n_heads = self.config.n_heads
        head_size = self.config.n_embeds // n_heads
        # Linear projections
        k = self.key(x).view(B, T, n_heads, head_size).transpose(1, 2)
        q = self.query(x).view(B, T, n_heads, head_size).transpose(1, 2)
        v = self.value(x).view(B, T, n_heads, head_size).transpose(1, 2)

        y = self._attend(x.shape, k, q, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.projection(y))
        return y


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, flash=True):
        super().__init__()
        assert config.n_embeds % config.n_heads == 0
        self.config = config
        self.flash = flash
        self.key = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.query = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.value = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.n_embeds, config.n_embeds, bias=config.bias)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.context_len, config.context_len))
        )

    def _attend(
        self, shape: torch.Size, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = shape
        if self.flash:
            dropout = self.config.dropout if self.training else 0.0
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True
            )
        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ v

    def forward(self, x):
        B, T, C = x.shape
        print(f"{x.shape=}")

        n_heads = self.config.n_heads
        head_size = self.config.n_embeds // n_heads
        # Linear projections
        key = self.key(x).view(B, T, n_heads, head_size).transpose(1, 2)
        query = self.query(x).view(B, T, n_heads, head_size).transpose(1, 2)
        value = self.value(x).view(B, T, n_heads, head_size).transpose(1, 2)

        # Sparse Attention
        outs = []
        for i in range(self.config.n_heads):
            start = i * head_size
            end = (i + 1) * head_size
            q = query[:, i, :, :]
            k = key[:, i, start:end, :]
            v = value[:, i, start:end, :]
            w = q @ k.transpose(-2, -1) * C**-0.5
            attn = torch.softmax(w, dim=-1)
            outs.append(attn @ v)
        outs = torch.cat(outs, dim=1).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(outs)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embeds, 4 * config.n_embeds),
            nn.GELU(),
            nn.Linear(4 * config.n_embeds, config.n_embeds),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class SABlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttentionV2(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embeds)
        self.ln2 = nn.LayerNorm(config.n_embeds)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


ModelForward = namedtuple(
    "ModelForward", "token_logits added_logits loss token_loss added_loss"
)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.n_tokens, config.n_embeds)
        self.position_embedding_table = nn.Embedding(
            config.context_len, config.n_embeds
        )
        self.track_position_embedding_table = nn.Embedding(
            config.n_track_pos_embeds, config.n_embeds
        )
        self.added_param_embed = nn.Linear(config.n_added_params, config.n_embeds)
        self.drop = nn.Dropout(config.dropout)
        self.attention = nn.Sequential(
            *[SABlock(config) for _ in range(config.n_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.n_embeds, bias=config.bias)
        self.token_prediction = nn.Linear(config.n_embeds, config.n_tokens, bias=False)
        self.added_prediction = nn.Linear(
            config.n_embeds, config.n_added_params, bias=False
        )
        tie_weights = torch.nn.Parameter(
            self.token_prediction.weight[: config.n_tokens, : config.n_embeds]
        )
        self.token_embedding_table.weight = tie_weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("projection.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _split_inputs(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """\
        Split into tokens, added params, track_position
        """
        B, T, _ = inputs.shape
        tokens = inputs[:, :, 0].long()
        added = inputs[:, :, 1 : 1 + self.config.n_added_params].view(
            B, T, self.config.n_added_params
        )
        track_pos = inputs[:, :, -1].view(B, T)
        track_pos = (self.config.n_track_pos_embeds * track_pos).long()
        return tokens, added, track_pos

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None) -> ModelForward:
        device = x.device
        B, T, C = x.shape
        assert T <= self.config.context_len, T
        tokens, added, track_pos = self._split_inputs(x)
        # print(f"{tokens.shape=}")
        # print(f"{added.shape=}")
        # print(f"{track_pos.shape=}")
        arange = torch.arange(T, dtype=torch.long, device=device)
        tok_emb = self.token_embedding_table(tokens)
        pos_emb = self.position_embedding_table(arange)
        track_pos_emb = self.track_position_embedding_table(track_pos)
        added_emb = self.added_param_embed(added)
        x = tok_emb + pos_emb + added_emb + track_pos_emb
        x = self.drop(x)
        x = self.attention(x)
        # x = torch.cat((x, added), dim=2)
        x = self.layer_norm(x)
        if targets is None:
            # only use last position for generation
            token_logits = self.token_prediction(x[:, [-1], :])
            added_logits = self.added_prediction(x[:, [-1], :])
            return ModelForward(token_logits, added_logits, None, None, None)
        token_logits = self.token_prediction(x)
        added_logits = self.added_prediction(x)
        B, T, C = token_logits.shape
        t_tokens, t_added, _ = self._split_inputs(targets)
        token_loss = F.cross_entropy(token_logits.view(B * T, C), t_tokens.view(B * T))
        # added_param_loss = F.mse_loss(added_logits, t_added, reduction="sum")
        # added_param_loss = F.l1_loss(added_logits, t_added)
        err = added_logits - t_added
        err[:, :, 1] *= 10  # amplify the delta loss
        sq_err = err**2
        mse = sq_err.mean(dim=1).mean(dim=0)
        added_param_loss = mse.sum()
        # added_param_loss = (err**2).mean()
        loss = token_loss + added_param_loss
        return ModelForward(
            token_logits, added_logits, loss, token_loss, added_param_loss
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        out = []
        for sample in idx[-1, :, : 1 + self.config.n_added_params].tolist():
            out.append([int(sample[0])] + sample[1:])

        for i in track(range(max_new_tokens)):
            idx = idx[:, -self.config.context_len :]  # truncate context
            forward = self(idx)
            token_logits = forward.token_logits[:, -1, :] / temperature
            if top_k is not None:
                val, _ = torch.topk(token_logits, min(top_k, token_logits.size(-1)))
                token_logits[token_logits < val[:, [-1]]] = float("-inf")

            probs = F.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # B, 1
            token = next_token.int().item()
            added_params = forward.added_logits[:, -1, :]
            next_track_pos = torch.tensor([[i / max_new_tokens]]).to(idx.device)
            # print(f"{idx.shape=}")
            # print(f"{next_token.shape=}")
            # print(f"{added_params.shape=}")
            # print(f"{next_track_pos.shape=}")
            next_idx = torch.cat((next_token, added_params, next_track_pos), dim=-1)
            # print(f"{next_idx.shape=}")
            out.append([token] + added_params.tolist()[0])

            idx = torch.cat(
                (
                    idx,
                    next_idx.view(1, -1, 1 + self.config.n_added_params + 1),
                ),
                dim=1,
            )
            if next_token.item() == END_SONG_TOKEN:
                print("track end before max new tokens")
                break
        return out
