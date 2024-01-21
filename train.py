from contextlib import contextmanager
import glob
from pprint import pp
import os.path
import sys

import cbor2
from rich.progress import track
import torch
import matplotlib.pyplot as plt
from torch.functional import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

from model import Model, ModelConfig

BATCH_SIZE = 128
RUN_KEY = "attention-v5"
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


def get_batch(data: torch.Tensor, config: ModelConfig, batch_size=BATCH_SIZE):
    ix = torch.randint(len(data) - config.context_len, (batch_size,))
    x = torch.stack([data[i : i + config.context_len] for i in ix])
    y = torch.stack([data[i + 1 : i + config.context_len + 1] for i in ix])
    return x.cuda(), y.cuda()


@contextmanager
def autocast_ctx():
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        yield


@torch.no_grad()
def estimate_loss(model: Model, train_data: Tensor, eval_data: Tensor, eval_iters=10):
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
    lossi = []
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    max_steps = 10_000
    for i in track(range(max_steps + 1), description="Training"):
        global_training_steps += 1
        Xb, Yb = get_batch(data, model.module.config)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits, loss, label_loss, added_param_loss = model(Xb, Yb)
        if i % 10 == 0:
            summary.add_scalar("loss", loss.item(), global_training_steps)
            summary.add_scalar("label_loss", label_loss.item(), global_training_steps)
            summary.add_scalar(
                "added_param_loss", added_param_loss.item(), global_training_steps
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        # track
        scheduler.step()
        if i % 100 == 0:
            print(
                f"{i:7d}/{max_steps}: {loss.item():.4f}, LR={scheduler.get_last_lr()[0]}"
            )
        lossi.append(loss.log10().item())
    plt.figure(figsize=(16, 6))
    r = (len(lossi) // 100) * 100
    plt.plot(torch.tensor(lossi)[:r].view(-1, 100).mean(1))
    plt.savefig("plot.svg", format="svg")
    return max_steps, lossi[-1]


def generate_samples(model: Model, sample_count=1, max_new_tokens=2_000):
    n_added = model.config.n_added_params
    ctx_len = model.config.context_len
    # create context with padding + song start token
    gen_ctx = torch.tensor(
        [[[1, *[0.0] * n_added]] * (ctx_len - 1) + [[2, *[0.0] * n_added]]],
        device="cuda",
    )
    for sample in range(sample_count):
        print(f"generating sample up to {max_new_tokens} tokens")
        generated = model.generate(gen_ctx.clone(), max_new_tokens=max_new_tokens)
        with open(f"sample-{sample}.tokens", "wb") as f:
            cbor2.dump(generated, f)
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


def main(args: list[str]):
    resume = len(sys.argv) > 1 and args[1] == "resume"
    generate = len(sys.argv) > 1 and args[1] == "generate"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    glob_path = "~/vmshare/midi/solo-piano/**/*.tokens"
    print(f"loading data from {glob_path}")
    track_paths = glob.glob(os.path.expanduser(glob_path))
    tracks = []
    token_vocab_size = 0
    for path in track_paths:
        with open(path, "rb") as f:
            header, track_data = cbor2.load(f)
            assert header["version"] == 3, "unexpected token file format"
            token_vocab_size = header["token_vocab_size"]
            if len(track_data) < 100:
                print("too short:", path)
                continue
            tracks.append(track_data)
    tracks = [t for t in tracks if t]

    data = torch.tensor([event for track in tracks for event in track])
    split_n = int(0.9 * len(data))
    train_data = data[:split_n]
    eval_data = data[split_n:]

    config = ModelConfig(n_tokens=token_vocab_size)
    model = Model(config).cuda()
    summary.add_graph(model, get_batch(train_data, config))

    stats = ModelStats(model)
    print(f"Model has {stats.num_parameters()} parameters.")

    estimate = estimate_loss(model, train_data, eval_data)
    pp(estimate)

    optimizer = get_optimizer(model.named_parameters(), learning_rate=1e-5)
    epoch_size = split_n // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10 * epoch_size, gamma=0.1)
    global_training_steps = 0
    if resume or generate:
        print("loading from checkpoint.pt")
        checkpoint = torch.load("checkpoint.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer_state = checkpoint["optimizer_state_dict"]
        # optimizer.load_state_dict(optimizer_state)
        global_training_steps = checkpoint["steps_trained"]
        if resume:
            model.train()
        if generate:
            model.eval()
    if not generate:
        para_model = nn.DataParallel(model)
        steps_trained, loss = train(
            para_model,
            train_data,
            optimizer,
            scheduler,
            global_training_steps=global_training_steps,
        )
        global_training_steps += steps_trained
        save(
            model,
            optimizer=optimizer,
            loss=loss,
            learning_rate=scheduler.get_last_lr()[0],
            steps_trained=global_training_steps,
        )
    generate_samples(model)


if __name__ == "__main__":
    main(sys.argv)
