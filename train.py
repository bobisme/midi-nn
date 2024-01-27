import argparse
from contextlib import contextmanager
from pprint import pp
import random

import cbor2
from rich.progress import track
import torch
import matplotlib.pyplot as plt
from torch.functional import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

from model import Model, ModelConfig

DEV = "cuda"

BATCH_SIZE = 128
RUN_KEY = "token-v4-model-v9"
summary = SummaryWriter(f"runs/{RUN_KEY}")

TrackElement = tuple[int, float, float, float, float]


class ModelStats:
    def __init__(self, model: Model):
        self.model = model

    def estimate_flops(self):
        pass

    def num_parameters(self):
        n_elements = sum(p.numel() for p in self.model.parameters())
        return n_elements


def get_batch(
    data: torch.Tensor, config: ModelConfig, batch_size=BATCH_SIZE, transpose_rate=0.0
):
    ix = torch.randint(len(data) - config.context_len, (batch_size,))
    x_data = torch.stack([data[i : i + config.context_len] for i in ix]).to(DEV)
    y_data = torch.stack([data[i + 1 : i + config.context_len + 1] for i in ix]).to(DEV)
    roll = random.random()
    if roll < transpose_rate:
        semis = random.choice(range(-5, 7))
        tokens = x_data[:, :, 0]
        x_data[:, :, 0] = torch.where(
            (5 < tokens) & (tokens < (5 + 2 * 127)), tokens + semis, tokens
        )
    return x_data, y_data


@contextmanager
def autocast_ctx():
    if DEV == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


@torch.no_grad()
def estimate_loss(model: Model, train_data: Tensor, eval_data: Tensor, eval_iters=10):
    print("estimating loss")
    out = {}
    model.eval()
    d = {"train": train_data, "eval": eval_data}
    for split in ("train", "eval"):
        data = d[split]
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, model.config)
            with autocast_ctx():
                _, loss, _, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_optimizer(
    named_params, learning_rate=1e-3, betas=(0.9, 0.95), weight_decay=0.1
):
    params = {name: param for name, param in named_params if param.requires_grad}
    decay_params = [p for p in params.values() if p.dim() >= 2]
    nodecay_params = [p for p in params.values() if p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=True)


def train(
    model: nn.DataParallel[Model],
    data,
    optimizer,
    scheduler,
    global_training_steps=0,
) -> tuple[int, float]:
    losses_log10 = []
    losses = []
    model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    max_steps = 20_000
    for i in track(range(max_steps + 1), description="Training"):
        global_training_steps += 1
        Xb, Yb = get_batch(data, model.module.config, transpose_rate=0.8)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            _, loss, label_loss, added_param_loss = model(Xb, Yb)
        if i % 10 == 0:
            summary.add_scalar("loss", loss.item(), global_training_steps)
            summary.add_scalar("label_loss", label_loss.item(), global_training_steps)
            summary.add_scalar(
                "added_param_loss", added_param_loss.item(), global_training_steps
            )
        scaled: Tensor = scaler.scale(loss)  # type: ignore
        scaled.backward()
        scaler.step(optimizer)
        scaler.update()
        # track
        scheduler.step()
        losses_log10.append(loss.log10().item())
        losses.append(loss.item())
        if i % 100 == 0:
            mean = torch.tensor(losses[-100:]).mean().item()
            last_lr = scheduler.get_last_lr()[0]
            print(f"{i:7d}/{max_steps}: {mean:.4f}, LR={last_lr}")
    plt.figure(figsize=(16, 6))
    r = (len(losses_log10) // 100) * 100
    plt.plot(torch.tensor(losses_log10)[:r].view(-1, 100).mean(1))
    plt.savefig("plot.svg", format="svg")
    return max_steps, losses_log10[-1]


@torch.no_grad()
def generate_samples(model: Model, args: argparse.Namespace, sample_count=1):
    model.eval()
    n_added = model.config.n_added_params
    ctx_len = model.config.context_len
    # create context with padding + song start token
    gen_ctx = torch.tensor(
        [[[1, *[0.0] * n_added, 0]] * (ctx_len - 1) + [[2, *[0.0] * n_added, 0]]],
        device=DEV,
    )
    for sample in range(sample_count):
        print(f"generating sample up to {args.max_new_tokens} tokens")
        generated = model.generate(
            gen_ctx.clone(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        header = {"version": 4, "token_vocab_size": 0}
        with open(f"sample-{sample}.tokens", "wb") as f:
            cbor2.dump((header, generated), f)  # type: ignore
            print("saved generated sample to", f.name)


def save(
    model: Model, optimizer, loss: float, learning_rate: float, steps_trained: int
):
    filename = f"{RUN_KEY}_{steps_trained}i.pt"
    save_data = {
        "model_state_dict": model.state_dict(),
        "steps_trained": steps_trained,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "learning_rate": learning_rate,
    }
    torch.save(save_data, filename)
    torch.save(save_data, "checkpoint.pt")
    print(f"saved checkpoint to {filename} and checkpoint.pt")


def load_tracks(track_paths: list[str]):
    tracks = []
    for path in track_paths:
        with open(path, "rb") as f:
            print(f"reading data from {f.name}")
            header, track_data = cbor2.load(f)  # type: ignore
            assert header["version"] == 4, "unexpected token file format"
            if len(track_data) < 100:
                print("too short:", path)
                continue
            for i, _ in enumerate(track_data):
                track_data[i].append(int(128.0 * i / len(track_data)))
            tracks.append(track_data)
    return [t for t in tracks if t]


def main(args: argparse.Namespace):
    if not (args.resume or args.generate or args.scratch):
        print("WHAT MODE SHALL I RUN?")
        exit(1)
    checkpoint_file = args.checkpoint or "checkpoint.pt"
    track_paths = args.files
    tracks = load_tracks(track_paths)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = torch.tensor([event for track in tracks for event in track])
    split_n = int(0.9 * len(data))
    train_data = data[:split_n]
    eval_data = data[split_n:]

    config = ModelConfig()
    model = Model(config).to(DEV)
    if args.scratch or args.resume:
        summary.add_graph(model, get_batch(train_data, config))

    stats = ModelStats(model)
    print(f"Model has {stats.num_parameters()} parameters.")

    if args.scratch:
        estimate = estimate_loss(model, train_data, eval_data)
        pp(estimate)

    global_training_steps = 0
    optimizer = None
    scheduler = None
    if not args.generate:
        optimizer = get_optimizer(model.named_parameters(), learning_rate=1e-3)
        epoch_size = split_n // BATCH_SIZE
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 20 * epoch_size, gamma=0.1
        )
    if args.resume or args.generate:
        print(f"LOADING from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not args.generate:
            optimizer_state = checkpoint["optimizer_state_dict"]
            assert optimizer
            optimizer.load_state_dict(optimizer_state)
            global_training_steps = checkpoint["steps_trained"]
    if not args.generate and (args.resume or args.scratch):
        para_model = nn.DataParallel(model)
        steps_trained, loss = train(
            para_model,
            train_data,
            optimizer,
            scheduler,
            global_training_steps=global_training_steps,
        )
        global_training_steps += steps_trained
        assert scheduler
        save(
            model,
            optimizer=optimizer,
            loss=loss,
            learning_rate=scheduler.get_last_lr()[0],
            steps_trained=global_training_steps,
        )
    generate_samples(model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train this thing.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for resuming training or generating",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument(
        "--generate", action="store_true", help="Just generate a new sample."
    )
    parser.add_argument("--scratch", action="store_true", help="Train from scratch")
    parser.add_argument(
        "files",
        metavar="FILE",
        type=str,
        nargs="*",
        help="an integer for the accumulator",
    )
    parser.add_argument("--cpu", action="store_true", help="CPU instead of cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2_000)
    args = parser.parse_args()
    main(args)
