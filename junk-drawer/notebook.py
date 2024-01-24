#%%
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

assert torch.cuda.is_available()
random.seed(42)
torch.manual_seed(42)

#%%
# CONSTANTS
CONTEXT_LENGTH = 8
BATCH_SIZE = 32
EMBED_LEN = 64

# %%
import cbor2
with open('embed-table-900-64.cbor', 'rb') as f:
    table = cbor2.load(f)
with open('moonlight.tokens', 'rb') as f:
    moonlight = cbor2.load(f)
#%%
VOCAB_SIZE =len(table)
#%%
print(moonlight[0:5])
print(len(moonlight))
#%%
def build_dataset(track):
    X, Y = [], []
    for event in track:
        # print(w)
        # context: list[int] = [[1, 0.0]] * CONTEXT_LENGTH
        context = [1] * CONTEXT_LENGTH
        for token, beats in track[:100]:
            # ix = stoi[ch]
            X.append(context)
            Y.append(torch.tensor([token, beats]))
            # print("".join(itos[i] for i in context), "-->", itos[ix])
            context = context[1:] + [torch.tensor([token, beats])]
            # context = context[1:] + [token]
            # context = context[1:] + [token, beats]
            print(context)
    return torch.tensor(X).cuda(), torch.tensor(Y).cuda()
#%%
track = moonlight
# training split, dev/validation split (for hyperparameters), test
# 80%, 10%, 10%
Xtrain, Ytrain = build_dataset(track[: int(0.8 * len(track))])
Xdev, Ydev = build_dataset(track[int(0.8 * len(track)) : int(0.9 * len(track))])
Xtest, Ytest = build_dataset(track[int(0.9 * len(track)) :])

#%%
print(Xtrain[:, :, 0])

#%% Modules

class Embedding:
    def __init__(self, num_embeddings, embeddings_dim) -> None:
        self.weight = torch.randn((num_embeddings, embeddings_dim), device="cuda")

    def __call__(self, IX):
        print(IX.shape)
        # self.out = self.weight[IX[:, :, 0]]
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]

class Linear:
    def __init__( self, fan_in: int, fan_out: int, bias=True, device="cuda"):
        self.device = device
        # Random initialization with kaiming-like normalization.
        self.weight = (
            torch.randn((fan_in, fan_out), device=device)
            / fan_in**0.5
        )
        self.bias = torch.zeros(fan_out, device=device) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    """\
    Batch Noemalization: Accelerating Deep Network Training by Reducing Internal
    Covariant Shift
    https://arxiv.org/abs/1502.03167

    Same as torch.nn.BatchNorm1d with:
        affine=True # will use beta and gamma
        track_running_stats=True
        device=whatever
        dtype=float
    """

    def __init__(self, dim, epsilon=1e-5, momentum=0.1, device="cuda"):
        self.device = device
        self.eps = epsilon
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
        # buffers, trained with running "momentum update" for smoothing
        self.running_mean = torch.zeros(dim, device=device)
        # tracking variance instead of stddev
        self.running_var = torch.ones(dim, device=device)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean  # µB from paper
            xvar = self.running_var  # σ^2B from paper
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        # used for statistic collection, not part of torch
        self.out = self.gamma * xhat + self.beta
        # update buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    """Same as torch.nn.Tanh"""

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []

class Sequential:
    def __init__(self, *layers) -> None:
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

#%% RUN
n_embed = EMBED_LEN
n_hidden = 128
model = Sequential(
    Embedding(VOCAB_SIZE, EMBED_LEN),
    Flatten(),
    Linear(n_embed * CONTEXT_LENGTH, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    # Linear(n_hidden * 2, n_hidden, bias=False),
    # BatchNorm1d(n_hidden),
    # Tanh(),
    # Linear(n_hidden * 2, n_hidden, bias=False),
    # BatchNorm1d(n_hidden),
    # Tanh(),
    Linear(n_hidden, VOCAB_SIZE),
)

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
max_steps = 40_000
batch_size = CONTEXT_LENGTH
lossi = []

for i in range(max_steps):
    ix = torch.randint(0, Xtrain.shape[0], (batch_size,))
    Xb, Yb = Xtrain[ix], Ytrain[ix]

    # forward
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # back
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 10_000 else 0.01 if i < 20_000 else 0.001
    for p in parameters:
        p.data += -lr * p.grad

    if i == 0:
        for layer in model.layers:
            print(f"{layer.__class__.__name__}: {tuple(layer.out.shape)}")

    # track
    if i % 1_000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())
#%%

plt.figure(figsize=(16, 6))
# plt.plot(lossi)
# instead, average across 1_000 points
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
# plt.savefig("plot.svg", format="svg")
#%%
for layer in model.layers:
    layer.training = False

@torch.no_grad()
def split_loss(split):
    x, y = {
        "train": (Xtrain, Ytrain),
        "eval": (Xdev, Ydev),
        "test": (Xtest, Ytest),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss("train")
split_loss("eval")
#%%
# sample
for sample in range(1):
    out = []
    context = [1] * CONTEXT_LENGTH
    for i in range(1000):
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        # shift the context window and track samples
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    with open(f"sample-{sample}.tokens", 'wb') as f:
        print(out)
        cbor2.dump(out, f)
        print("wrote", f.name)
# %%
