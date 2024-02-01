#!/usr/bin/env python
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pprint import pp
import random
from typing import NamedTuple

import cbor2
from rich.progress import track
import torch
import matplotlib.pyplot as plt
from torch.functional import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from model import Model, ModelConfig, ModelForward

DEV = "cuda"

TOKEN_VERSION = 5
MODEL_VERSION = 11
DEFAULT_BATCH_SIZE = 12
RUN_KEY = f"token-v{TOKEN_VERSION}-model-v{MODEL_VERSION}"
summary = SummaryWriter(f"runs/{RUN_KEY}")

TrackElement = tuple[int, float, float, float, float]


Data = NamedTuple("Data", [("train", Tensor), ("eval", Tensor), ("split_n", int)])


class ModelStats:
    def __init__(self, model: Model):
        self.model = model

    def estimate_flops(self):
        pass

    def num_parameters(self):
        n_elements = sum(p.numel() for p in self.model.parameters())
        return n_elements


def get_batch(
    data: torch.Tensor,
    config: ModelConfig,
    batch_size=DEFAULT_BATCH_SIZE,
    transpose_rate=0.0,
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
def estimate_loss(model: Model, data: Data, eval_iters=10):
    out = {}
    model.eval()
    for split in ("train", "eval"):
        d = getattr(data, split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(d, model.config)
            with autocast_ctx():
                forward = model(X, Y)
            losses[k] = forward.loss.item()
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


@dataclass
class TrainOptions:
    model: nn.Module
    data: Data
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    batch_size: int = DEFAULT_BATCH_SIZE
    max_steps: int = 20_000
    global_training_steps: int = 0
    transpose_rate: float = 0.8


TrainResult = NamedTuple("TrainResult", [("steps", int), ("log_loss", float)])


def train(opts: TrainOptions) -> TrainResult:
    losses_log10 = []
    losses = []
    opts.model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    print(
        f"training {opts.max_steps} @ batch size {opts.batch_size} with config:\n{opts.model.module.config}"
    )

    estimate = estimate_loss(opts.model.module, opts.data)
    pp(estimate)
    last_eval = estimate["eval"]

    for i in track(range(opts.max_steps + 1), description="Training"):
        opts.global_training_steps += 1
        Xb, Yb = get_batch(
            opts.data.train, opts.model.module.config, transpose_rate=0.0
        )
        opts.optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            forward: ModelForward = opts.model(Xb, Yb)
        if i % 10 == 0:
            summary.add_scalar(
                "learning_rate",
                opts.scheduler.get_last_lr()[0],
                opts.global_training_steps,
            )
            summary.add_scalar("loss", forward.loss.item(), opts.global_training_steps)
            summary.add_scalar(
                "token_loss", forward.token_loss.item(), opts.global_training_steps
            )
            summary.add_scalar(
                "added_param_loss",
                forward.added_loss.item(),
                opts.global_training_steps,
            )
        scaled: Tensor = scaler.scale(forward.loss)  # type: ignore
        scaled.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(opts.model.parameters(), 1.0)
        scaler.step(opts.optimizer)
        scaler.update()
        # track
        opts.scheduler.step()
        losses_log10.append(forward.loss.log10().item())
        losses.append(forward.loss.item())
        if i % 100 == 0:
            mean = torch.tensor(losses[-100:]).mean().item()
            last_lr = opts.scheduler.get_last_lr()[0]
            estimated_loss = estimate_loss(opts.model.module, opts.data, eval_iters=1)
            last_eval = estimated_loss["eval"]
            print(
                f"{i:7d}/{opts.max_steps}: {mean:.4f}, eval_loss={last_eval:.4f} LR={last_lr:.8f}"
            )
            summary.add_scalar(
                "eval_loss",
                last_eval.item(),
                opts.global_training_steps,
            )
    plt.figure(figsize=(16, 6))
    r = (len(losses_log10) // 100) * 100
    plt.plot(torch.tensor(losses_log10)[:r].view(-1, 100).mean(1))
    plt.savefig("plot.svg", format="svg")
    return TrainResult(opts.max_steps, losses_log10[-1])


@torch.no_grad()
def generate_samples(model: Model, args: argparse.Namespace, sample_count=1):
    model.eval()
    n_added = model.config.n_added_params
    ctx_len = model.config.context_len
    # gen_ctx = torch.tensor([[[2, *[0.0] * n_added, 0]]], device=DEV)
    # WE MUST OVERFIT
    tracks = load_tracks(["tokenized/mond_3_format0.mid.tokens"])
    data = load_data(tracks)
    starter_len = 100
    gen_ctx = data.train[:starter_len].view(1, starter_len, 1 + n_added + 1).to(DEV)
    print(f"{gen_ctx.shape=}")

    for sample in range(sample_count):
        print(f"generating sample up to {args.max_new_tokens} tokens")
        generated = model.generate(
            gen_ctx.clone(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        header = {"version": TOKEN_VERSION, "token_vocab_size": 0}
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
            assert header["version"] == TOKEN_VERSION, "unexpected token file format"
            if len(track_data) < 100:
                print("too short:", path)
                continue
            for i, _ in enumerate(track_data):
                track_data[i].append(i / len(track_data))
            tracks.append(track_data)
    return [t for t in tracks if t]


def load_data(tracks, split_ratio=0.9) -> Data:
    data = torch.tensor([event for track in tracks for event in track])
    split_n = int(split_ratio * len(data))
    train_data = data[:split_n]
    eval_data = data[split_n:]
    return Data(train_data, eval_data, split_n)


def main(args: argparse.Namespace):
    cmd = args.command
    checkpoint_file = args.checkpoint or "checkpoint.pt"
    data = Data(None, None, 0)
    if cmd in ("scratch", "resume"):
        track_paths = args.files
        tracks = load_tracks(track_paths)
        data = load_data(tracks)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = ModelConfig()
    model = Model(config).to(DEV)
    # if args.scratch or args.resume:
    #     summary.add_graph(model, get_batch(train_data, config))

    stats = ModelStats(model)
    print(f"Model has {stats.num_parameters()} parameters.")

    global_training_steps = 0
    lr = 1e-3 if getattr(args, "learning_rate", None) is None else args.learning_rate
    optimizer = get_optimizer(model.named_parameters(), learning_rate=lr)
    epoch_size = data.split_n // getattr(args, "batch_size", 1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2 * epoch_size, gamma=0.8)

    if cmd in ("resume", "generate"):
        print(f"LOADING from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not cmd == "generate":
            optimizer_state = checkpoint["optimizer_state_dict"]
            assert optimizer
            if args.learning_rate is None:
                optimizer.load_state_dict(optimizer_state)
            global_training_steps = checkpoint["steps_trained"]

    if cmd in ("scratch", "resume"):
        # opt_model = torch.compile(model)
        para_model = nn.DataParallel(model)
        steps_trained, loss = train(
            TrainOptions(
                model=para_model,
                data=data,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_size=args.batch_size,
                max_steps=args.iterations,
                global_training_steps=global_training_steps,
            )
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
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Just generate a new sample.")
    scratch = subparsers.add_parser("scratch", help="Train from scratch")
    resume = subparsers.add_parser("resume", help="Resume training")

    for p in (generate, scratch, resume):
        p.add_argument("--cpu", action="store_true", help="CPU instead of cuda")
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--top-k", type=int, default=None)
        p.add_argument("--max-new-tokens", type=int, default=2_000)

    for p in (generate, resume):
        p.add_argument("--checkpoint", type=str, help="Path to checkpoint file")

    for p in (scratch, resume):
        p.add_argument(
            "files",
            metavar="FILE",
            type=str,
            nargs="*",
            help="token files",
        )
        p.add_argument("--learning-rate", type=float, default=None)
        p.add_argument("--iterations", type=int, default=20_000)
        p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()
    main(args)
