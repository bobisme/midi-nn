import math
from dataclasses import dataclass

from rich.progress import track
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available()


@dataclass
class ModelConfig:
    context_len: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_embeds: int = 32
    n_tokens: int = 519
    n_added_params: int = 2
    dropout: float = 0.0
    bias: bool = True

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
            "tril", torch.tril(torch.ones(config.context_len, config.context_len))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
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


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embeds, 4 * config.n_embeds),
            nn.ReLU(),  # TODO: GELU
            nn.Linear(4 * config.n_embeds, config.n_embeds),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class SABlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.n_embeds // config.n_heads
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embeds)
        self.ln2 = nn.LayerNorm(config.n_embeds)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.n_tokens, config.n_embeds)
        self.position_embedding_table = nn.Embedding(
            config.context_len, config.n_embeds
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[SABlock(config) for _ in range(config.n_layers)])
        self.layer_norm = nn.LayerNorm(config.n_embed_and_added, bias=config.bias)
        self.lm_head = nn.Linear(
            config.n_embed_and_added, config.n_tokens_and_added, bias=False
        )
        # Tie weights
        print(f"{self.token_embedding_table.weight.shape=}")
        print(f"{self.lm_head.weight.shape=}")
        tie_weights = torch.nn.Parameter(
            self.lm_head.weight[: config.n_tokens, : config.n_embeds]
        )
        print(f"{tie_weights.shape=}")
        self.token_embedding_table.weight = tie_weights
        self.apply(self._init_weights)
        # Scale the weights of residual layers at initialization by a factor of 1/√N
        # where N is the number of residual layers.
        # Language Models are Unsupervised Multitask Learners
        for name, p in self.named_parameters():
            if name.endswith("projection.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        B, T, C = x.shape
        assert T <= self.config.context_len, T
        tokens = x[:, :, 0].long()
        added = x[:, :, 1:].view(
            -1, self.config.context_len, self.config.n_added_params
        )
        tok_emb = self.token_embedding_table(tokens)
        pos_emb = self.position_embedding_table(
            torch.arange(T, dtype=torch.long, device=device)
        )
        x = tok_emb + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = torch.cat((x, added), dim=2)
        x = self.layer_norm(x)
        if targets is None:
            # only use last position for generation
            logits = self.lm_head(x[:, [-1], :])
            return logits, None, None, None
        logits = self.lm_head(x)
        B, T, C = logits.shape
        classes = targets[:, :, 0].long()
        added_params = targets[:, :, 1:]
        label_loss = F.cross_entropy(logits.view(B * T, C), classes.view(B * T))
        sq_err = (
            added_params[:, :, -self.config.n_added_params :]
            - logits[:, :, -self.config.n_added_params :]
        ) ** 2
        added_param_loss = sq_err.mean(dim=1).mean(dim=0).sum()
        loss = label_loss + added_param_loss
        return logits, loss, label_loss, added_param_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        out = []
        for _ in track(range(max_new_tokens)):
            idx = idx[:, -self.config.context_len :]  # truncate context
            logits, _, _, _ = self(idx)
            # TODO: temperature and top-k
            probs = F.softmax(logits[:, -1, : self.config.n_tokens], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # B, 1
            token = next_token.int().item()
            added_params = logits[:, -1, -self.config.n_added_params :].view(
                1, self.config.n_added_params
            )
            next_idx = torch.cat((next_token, added_params), dim=-1)  # B, 3
            out.append([token] + added_params.tolist()[0])
            idx = torch.cat(
                (idx, next_idx.view(1, -1, 1 + self.config.n_added_params)), dim=1
            )
            if next_token.item() == 3:
                print("track end before max new tokens")
                break
        return out
