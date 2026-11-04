#!/usr/bin/env python3
"""CIFAR-10 image classification with PyTorch.

Trains a small convolutional network with a reproducible, fully logged workflow:
data loading + augmentation, a train/validation/test split, per-epoch validation,
best-model checkpointing, a confusion matrix, and experiment tracking.

Usage
-----
Single process (CPU or one GPU):
    python train.py --epochs 5 --batch-size 128 --lr 1e-3

Several GPUs on one machine (one process per GPU):
    torchrun --nproc_per_node=2 train.py --epochs 5 --batch-size 128 --lr 1e-3 --tracker wandb

Through the launcher (GPU detection, validation, logging, troubleshooting hints):
    bash run_distributed.sh --epochs 5 --batch-size 128

Everything a run produces is written to <output-dir>/<run-name>/:
    best_model.pth / last_model.pth   model state_dicts (best = highest validation accuracy)
    metrics.csv                       per-epoch loss/accuracy table
    run_config.json                   arguments, hardware, versions, determinism flags
    final_metrics.json                test-set metrics (accuracy, per-class precision/recall/F1)
    confusion_matrix.csv / .png       test-set confusion matrix
    requirements.txt                  package versions used for the run
"""
import argparse
import csv
import glob
import importlib.util
import io
import json
import os
import random
import sys
import time
from datetime import datetime
from importlib import metadata as md
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision.transforms import v2

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
NUM_CLASSES = len(CLASSES)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# =============================================================================
# Command line interface
# =============================================================================
def int_in_range(lo, hi):
    """argparse type: integer within [lo, hi], with a readable error message."""
    def convert(text):
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}")
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"expected an integer from {lo} to {hi}, got {value}")
        return value
    return convert


def float_in_range(lo, hi, lo_open=False):
    """argparse type: float within [lo, hi] (lower bound optionally exclusive)."""
    def convert(text):
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a number, got {text!r}")
        if not (lo < value if lo_open else lo <= value) or value > hi:
            bound = f"> {lo}" if lo_open else f">= {lo}"
            raise argparse.ArgumentTypeError(f"expected a number {bound} and <= {hi}, got {value}")
        return value
    return convert


def build_parser():
    """Defines every option. Defaults are chosen to give a sensible first run out of the box."""
    # allow_abbrev=False: a mistyped/shortened flag is an error instead of silently matching another one
    p = argparse.ArgumentParser(description="CIFAR-10 CNN training", allow_abbrev=False,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int_in_range(1, 100000), default=5, help="passes over the training set")
    g.add_argument("--batch-size", type=int_in_range(1, 1000000), default=128,
                   help="mini-batch size PER PROCESS (global batch = batch-size x number of GPUs)")
    g.add_argument("--lr", type=float_in_range(0.0, 10.0, lo_open=True), default=1e-3, help="learning rate")
    g.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="optimisation algorithm")
    g.add_argument("--momentum", type=float_in_range(0.0, 1.0), default=0.9, help="momentum (used by sgd only)")
    g.add_argument("--val-split", type=float_in_range(0.0, 0.5, lo_open=True), default=0.1,
                   help="fraction of the training set held out for validation / best-model selection")
    g.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="random crop + horizontal flip on the training images (--no-augment disables)")
    g.add_argument("--log-interval", type=int_in_range(1, 10**9), default=100, help="print progress every N mini-batches")

    g = p.add_argument_group("data and outputs")
    g.add_argument("--data", "--data-dir", dest="data", default="./data",
                   help="directory holding (or receiving) the CIFAR-10 parquet files")
    g.add_argument("--output-dir", default="./outputs",
                   help="parent directory; each run writes to <output-dir>/<run-name>/")
    g.add_argument("--num-workers", type=int_in_range(0, 256), default=2, help="DataLoader worker processes per process")

    g = p.add_argument_group("reproducibility")
    g.add_argument("--seed", type=int_in_range(0, 2**32 - 1), default=42, help="seed for every random number generator")
    g.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                   help="deterministic kernels/cuDNN settings (slower; --no-deterministic disables)")

    g = p.add_argument_group("device")
    g.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="auto = $RUN_DEVICE if set, else CUDA when available, else CPU")

    g = p.add_argument_group("experiment tracking")
    g.add_argument("--tracker", choices=["wandb", "mlflow", "none"], default="wandb",
                   help="metrics/params/artifact tracker (local CSV/JSON logs are always written)")
    g.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="Weights & Biases mode (only used with --tracker wandb)")
    g.add_argument("--project", default="cifar10-cnn", help="W&B project / MLflow experiment name")
    g.add_argument("--run-name", default=None, help="run label; default: $RUN_ID, else a timestamp")
    return p


# =============================================================================
# Distributed context and device selection
# =============================================================================
class DistContext:
    """Process-group information. torchrun sets RANK / WORLD_SIZE / LOCAL_RANK for every
    process it starts; when they are absent we are a plain single process (world size 1)."""

    def __init__(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world_size > 1
        self.is_main = self.rank == 0          # only the main process prints, saves and tracks
        self.device = torch.device("cpu")

    def log(self, *message):
        if self.is_main:
            print(*message, flush=True)

    def barrier(self):
        if self.enabled:
            dist.barrier(device_ids=[self.local_rank] if self.device.type == "cuda" else None)

    def all_reduce_sum(self, tensor):
        """Sums a tensor across all processes (no-op for a single process)."""
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor


def pick_device(requested):
    """Resolves --device / $RUN_DEVICE; fails loudly if CUDA is demanded but unusable."""
    if requested == "auto":
        # the launcher exports RUN_DEVICE after probing the hardware
        requested = os.environ.get("RUN_DEVICE", "").strip().lower() or "auto"
        if requested not in ("auto", "cuda", "cpu"):
            print(f"[device] ignoring unrecognised RUN_DEVICE={requested!r}")
            requested = "auto"
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: CUDA was requested but torch.cuda.is_available() is False "
                 "(no GPU, CPU-only PyTorch build, or driver mismatch). Use --device cpu to run on the CPU.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed(device_request):
    """Chooses the device for this process and joins the process group when launched by torchrun."""
    ctx = DistContext()
    device = pick_device(device_request)
    if ctx.enabled:
        if device.type == "cuda":
            if ctx.local_rank >= torch.cuda.device_count():
                sys.exit(f"ERROR: local rank {ctx.local_rank} has no GPU ({torch.cuda.device_count()} visible). "
                         "Launch with --nproc_per_node <= the number of GPUs.")
            torch.cuda.set_device(ctx.local_rank)             # one GPU per process
            device = torch.device("cuda", ctx.local_rank)
            backend = "nccl"                                  # fastest GPU-to-GPU communication
        else:
            backend = "gloo"                                  # CPU fallback backend
        dist.init_process_group(backend=backend)
        ctx.backend = backend
    ctx.device = device
    return ctx


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed, deterministic):
    """Seeds every RNG in use and configures deterministic kernels. Returns the flags applied,
    so they can be recorded alongside the run."""
    os.environ["PYTHONHASHSEED"] = str(seed)                  # only affects subprocesses
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                          # lazy: harmless without CUDA
    if deterministic:
        # must be set before the first cuBLAS call for deterministic GPU matmuls
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)   # warn (not crash) on ops without a deterministic version
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True                 # fastest kernels, non-reproducible
    return {"seed": seed,
            "use_deterministic_algorithms": deterministic,
            "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def seed_worker(worker_id):
    """Gives each DataLoader worker a distinct but reproducible seed (numpy/random are not
    re-seeded by PyTorch automatically)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def write_requirements(path):
    """Pins the package versions of this environment so the run can be recreated."""
    packages = ("torch", "torchvision", "numpy", "pandas", "pyarrow", "pillow", "matplotlib",
                "huggingface_hub", "wandb", "mlflow")
    lines = [f"# python {sys.version.split()[0]} | torch built for CUDA {torch.version.cuda}"]
    for pkg in packages:
        try:
            installed = md.version(pkg)
            pinned = installed.split("+")[0]        # drop local build tags such as "+cu130": PyPI does not host them,
            note = f"  # installed build: {installed}" if pinned != installed else ""   # so the file stays installable
            lines.append(f"{pkg}=={pinned}{note}")
        except md.PackageNotFoundError:
            lines.append(f"# {pkg}: not installed")
    Path(path).write_text("\n".join(lines) + "\n")
    return lines


# =============================================================================
# Data
# =============================================================================
def find_parquet(data_dir, split):
    """Returns the first '<split>-*.parquet' file below data_dir, or None."""
    files = sorted(glob.glob(f"{data_dir}/**/{split}-*.parquet", recursive=True))
    return files[0] if files else None


def ensure_dataset(data_dir, ctx):
    """Downloads CIFAR-10 (Hugging Face hub) once, on the main process only; others wait."""
    if ctx.is_main:
        if find_parquet(data_dir, "train") and find_parquet(data_dir, "test"):
            ctx.log(f"CIFAR-10 already present under {data_dir}; skipping download.")
        else:
            from huggingface_hub import snapshot_download
            ctx.log(f"Downloading CIFAR-10 to {data_dir} ...")
            snapshot_download(repo_id="uoft-cs/cifar10", repo_type="dataset", cache_dir=data_dir)
    ctx.barrier()
    for split in ("train", "test"):
        if not find_parquet(data_dir, split):
            sys.exit(f"ERROR: No {split} parquet found under {data_dir}. Check --data and that the download finished.")


class CifarParquet(Dataset):
    """CIFAR-10 held in memory as PNG bytes (Hugging Face parquet format), decoded lazily per item."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]["bytes"])).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_transforms(augment):
    """Training pipeline (optional augmentation) and a deterministic evaluation pipeline."""
    normalise = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(CIFAR_MEAN, CIFAR_STD)]
    augmentation = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()] if augment else []
    return v2.Compose(augmentation + normalise), v2.Compose(normalise)


def shard(dataset, ctx):
    """Gives each process a disjoint, strided slice so evaluation sees every sample exactly once
    (DistributedSampler would pad with duplicates and slightly bias the metrics)."""
    if not ctx.enabled:
        return dataset
    return Subset(dataset, list(range(ctx.rank, len(dataset), ctx.world_size)))


def build_data(args, ctx):
    """Builds train / validation / test datasets and their loaders."""
    train_tf, eval_tf = build_transforms(args.augment)
    train_df = pd.read_parquet(find_parquet(args.data, "train"))
    test_df = pd.read_parquet(find_parquet(args.data, "test"))
    images, labels = train_df["img"].tolist(), train_df["label"].tolist()

    # Seeded split, independent of the number of processes. Two views of the same samples:
    # the training view is augmented, the validation view is not.
    order = np.random.RandomState(args.seed).permutation(len(labels))
    n_val = max(1, int(round(len(labels) * args.val_split)))
    val_idx, train_idx = order[:n_val].tolist(), order[n_val:].tolist()
    train_set = Subset(CifarParquet(images, labels, train_tf), train_idx)
    val_set = Subset(CifarParquet(images, labels, eval_tf), val_idx)
    test_set = CifarParquet(test_df["img"].tolist(), test_df["label"].tolist(), eval_tf)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  worker_init_fn=seed_worker, pin_memory=(ctx.device.type == "cuda"))  # pinned memory speeds up host->GPU copies
    if ctx.enabled:
        # each process sees a different shard of the training data each epoch
        sampler = DistributedSampler(train_set, num_replicas=ctx.world_size, rank=ctx.rank,
                                     shuffle=True, seed=args.seed)
        train_loader = DataLoader(train_set, sampler=sampler, **common)
    else:
        sampler = None
        generator = torch.Generator().manual_seed(args.seed)    # makes the shuffle order reproducible
        train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(shard(val_set, ctx), shuffle=False, **common)
    test_loader = DataLoader(shard(test_set, ctx), shuffle=False, **common)
    sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_set)}
    return train_loader, val_loader, test_loader, sampler, test_set, sizes


# =============================================================================
# Model
# =============================================================================
class SimpleCNN(nn.Module):
    """Three conv blocks (conv -> batch-norm -> ReLU -> max-pool) followed by a two-layer classifier.

    Input : float tensor (N, 3, 32, 32), normalised CIFAR-10 images
    Output: float tensor (N, 10), unnormalised class scores (logits)
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv2, self.bn2 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv3, self.bn3 = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)                         # halves height and width
        self.dropout = nn.Dropout(dropout)                     # regularisation for the classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """Maps a batch of images (N, 3, 32, 32) to class logits (N, num_classes)."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))         # -> (N, 32, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))         # -> (N, 64, 8, 8)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))         # -> (N, 128, 4, 4)
        x = torch.flatten(x, 1)                                # -> (N, 2048)
        x = self.dropout(F.relu(self.fc1(x)))                  # -> (N, 256)
        return self.fc2(x)                                     # -> (N, num_classes)


def model_summary(model):
    """Layer-by-layer parameter counts; returns the total number of trainable parameters."""
    lines, total = [], 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total += n
        if n:
            lines.append(f"  {name:<8}{type(module).__name__:<12}{n:>10,}")
    return total, "\n".join(lines)


def build_optimizer(args, model):
    """Adam or SGD(+momentum). Must be created after the model is on its final device."""
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


# =============================================================================
# Training and evaluation
# =============================================================================
def to_device(batch, device):
    """Moves an (images, labels) batch to the model's device (asynchronously when memory is pinned)."""
    inputs, labels = batch
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def sync_buffers(model, ctx):
    """Copies rank 0's BatchNorm running statistics to every process, so all processes evaluate
    (and checkpoint) the same model. Parameters are already identical thanks to DDP."""
    if ctx.enabled:
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


@torch.no_grad()
def evaluate(model, loader, criterion, ctx):
    """Evaluates on a loader without tracking gradients. Returns loss, accuracy (%) and the
    confusion matrix (rows = true class, columns = predicted class), aggregated over all processes."""
    device = ctx.device
    was_training = model.training
    model.eval()                                               # disables dropout, uses BatchNorm running stats
    totals = torch.zeros(3, device=device)                     # [loss sum, correct, samples]
    confusion = torch.zeros(NUM_CLASSES * NUM_CLASSES, dtype=torch.long, device=device)
    for batch in loader:
        inputs, labels = to_device(batch, device)
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        totals[0] += criterion(outputs, labels) * labels.size(0)   # criterion gives a batch mean: weight by batch size
        totals[1] += (predictions == labels).sum()
        totals[2] += labels.size(0)
        confusion += torch.bincount(labels * NUM_CLASSES + predictions, minlength=NUM_CLASSES ** 2)
    ctx.all_reduce_sum(totals)
    ctx.all_reduce_sum(confusion)
    model.train(was_training)
    return {"loss": (totals[0] / totals[2]).item(), "acc": (100 * totals[1] / totals[2]).item(),
            "confusion": confusion.view(NUM_CLASSES, NUM_CLASSES).cpu().numpy()}


def train_one_epoch(model, loader, optimizer, criterion, ctx, epoch, args, tracker, step):
    """Runs one pass over the training data. Returns (mean loss, accuracy %, updated global step)."""
    device = ctx.device
    model.train()
    # Statistics stay on the device as tensors; reading them with .item() every step would
    # force a GPU->CPU synchronisation per mini-batch.
    window = torch.zeros(3, device=device)                     # [loss sum, correct, samples] since last print
    total = torch.zeros(3, device=device)                      # same, for the whole epoch
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        inputs, labels = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)                  # 1. clear old gradients
        outputs = model(inputs)                                # 2. forward pass
        loss = criterion(outputs, labels)                      # 3. loss
        loss.backward()                                        # 4. backpropagation
        optimizer.step()                                       # 5. parameter update
        step += 1

        n = labels.size(0)
        batch_loss = loss.detach() * n                         # loss is a batch mean: weight by batch size
        batch_correct = (outputs.argmax(dim=1) == labels).sum()
        for stats in (window, total):
            stats[0] += batch_loss
            stats[1] += batch_correct
            stats[2] += n

        if (i + 1) % args.log_interval == 0 or (i + 1) == n_batches:
            reduced = ctx.all_reduce_sum(window.clone())       # same iteration on every rank, so the collective matches up
            loss_avg, acc_avg = (reduced[0] / reduced[2]).item(), (100 * reduced[1] / reduced[2]).item()
            ctx.log(f"[epoch {epoch}/{args.epochs} | batch {i + 1}/{n_batches} ({100 * (i + 1) // n_batches}%)] "
                    f"loss {loss_avg:.4f}  acc {acc_avg:.2f}%")
            tracker.log_metrics({"train/step_loss": loss_avg, "train/step_acc": acc_avg, "epoch": epoch}, step)
            window.zero_()

    ctx.all_reduce_sum(total)
    return (total[0] / total[2]).item(), (100 * total[1] / total[2]).item(), step


def save_state_dict(model, path):
    """Saves CPU copies of the weights so the file loads on any machine, with or without a GPU."""
    module = model.module if isinstance(model, DDP) else model
    torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, path)


def per_class_report(confusion):
    """Precision / recall / F1 for each class, derived from a confusion matrix."""
    tp = np.diag(confusion).astype(float)
    precision = tp / np.maximum(confusion.sum(axis=0), 1)      # of everything predicted as c, how much was c
    recall = tp / np.maximum(confusion.sum(axis=1), 1)         # of everything that is c, how much was found
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {c: {"precision": precision[i], "recall": recall[i], "f1": f1[i],
                "support": int(confusion[i].sum())} for i, c in enumerate(CLASSES)}


def save_confusion_matrix(confusion, directory):
    """Writes the confusion matrix as CSV (always) and as an annotated heat-map PNG (if matplotlib is available)."""
    csv_path = directory / "confusion_matrix.csv"
    pd.DataFrame(confusion, index=[f"true_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(csv_path)
    png_path = directory / "confusion_matrix.png"
    try:
        import matplotlib
        matplotlib.use("Agg")                                  # headless backend: no display needed
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES), CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Test-set confusion matrix")
        threshold = confusion.max() / 2
        for r in range(NUM_CLASSES):
            for c in range(NUM_CLASSES):
                ax.text(c, r, int(confusion[r, c]), ha="center", va="center", fontsize=8,
                        color="white" if confusion[r, c] > threshold else "black")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as e:                                     # plotting must never fail a finished training run
        print(f"(confusion matrix image skipped: {type(e).__name__}: {e})")
        png_path = None
    return csv_path, png_path


def inference_demo(model, dataset, device, n=8):
    """Classifies the first n test images with the (re)loaded model and prints the predictions."""
    model.eval()
    images = torch.stack([dataset[i][0] for i in range(n)])
    truth = [dataset[i][1] for i in range(n)]
    with torch.no_grad():
        probabilities = F.softmax(model(images.to(device)), dim=1).cpu()
    confidence, predicted = probabilities.max(dim=1)
    print("Inference with the reloaded best model:")
    print(f"  {'#':<3}{'true':<8}{'predicted':<11}{'confidence':>10}")
    for i in range(n):
        mark = "ok" if predicted[i].item() == truth[i] else "WRONG"
        print(f"  {i:<3}{CLASSES[truth[i]]:<8}{CLASSES[predicted[i]]:<11}{100 * confidence[i].item():>9.1f}%  {mark}")


# =============================================================================
# Experiment tracking
# =============================================================================
class Tracker:
    """No-op tracker; also the interface the real trackers implement. Using it for
    '--tracker none' (or when a tracker is unavailable) keeps the training code free of 'if tracker:' checks."""

    def log_metrics(self, metrics, step):
        pass

    def log_summary(self, summary):
        pass

    def log_image(self, key, path):
        pass

    def log_outputs(self, model_files, report_files):
        pass

    def finish(self):
        pass


class WandbTracker(Tracker):
    """Weights & Biases: config, metrics, summary values, plots and model artifacts."""

    def __init__(self, args, config, run_name):
        import wandb
        self.wandb = wandb
        if args.wandb_mode == "online" and "WANDB_API_KEY" not in os.environ:   # offline mode needs no key
            # fall back to a Kaggle secret when running there
            try:
                from kaggle_secrets import UserSecretsClient
                os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
            except Exception as e:
                print(f"No WANDB_API_KEY found ({type(e).__name__}).")
        try:
            self.run = wandb.init(project=args.project, name=run_name, config=config, mode=args.wandb_mode)
        except Exception as e:                                 # no internet / missing or invalid key
            print(f"W&B init failed ({type(e).__name__}: {e}); using offline mode "
                  "(upload later with: wandb sync wandb/offline-run-*).")
            self.run = wandb.init(project=args.project, name=run_name, config=config, mode="offline")

    def log_metrics(self, metrics, step):
        self.run.log(metrics, step=step)

    def log_summary(self, summary):
        for key, value in summary.items():
            self.run.summary[key] = value

    def log_image(self, key, path):
        self.run.log({key: self.wandb.Image(str(path))})

    def log_outputs(self, model_files, report_files):
        model = self.wandb.Artifact(f"model-{self.run.id}", type="model")
        for f in model_files:
            model.add_file(str(f))
        self.run.log_artifact(model)
        reports = self.wandb.Artifact(f"reports-{self.run.id}", type="evaluation")
        for f in report_files:
            reports.add_file(str(f))
        self.run.log_artifact(reports)

    def finish(self):
        self.run.finish()


class MlflowTracker(Tracker):
    """MLflow: parameters, metrics and the model/report files. Data is stored under <output-dir>
    (a SQLite database plus an artifact folder) unless MLFLOW_TRACKING_URI points elsewhere."""

    def __init__(self, args, config, run_name):
        import mlflow
        self.mlflow = mlflow
        out = Path(args.output_dir).resolve()
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            if importlib.util.find_spec("sqlalchemy"):          # MLflow's SQL backend needs SQLAlchemy
                uri = f"sqlite:///{out / 'mlflow.db'}"
            else:
                # MLflow 3 disables the plain-folder backend unless explicitly allowed
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
                uri = (out / "mlruns").as_uri()
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(args.project) is None:
            # keep artifacts next to the other outputs instead of a stray ./mlruns folder
            mlflow.create_experiment(args.project, artifact_location=(out / "mlartifacts").as_uri())
        mlflow.set_experiment(args.project)
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: str(v)[:250] for k, v in config.items()})   # MLflow limits parameter values to 250 chars
        print(f"MLflow tracking URI: {uri}")

    def log_metrics(self, metrics, step):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_summary(self, summary):
        self.mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, (int, float))})

    def log_image(self, key, path):
        self.mlflow.log_artifact(str(path), artifact_path="evaluation")

    def log_outputs(self, model_files, report_files):
        for f in model_files:
            self.mlflow.log_artifact(str(f), artifact_path="model")
        for f in report_files:
            self.mlflow.log_artifact(str(f), artifact_path="evaluation")

    def finish(self):
        self.mlflow.end_run()


def make_tracker(args, ctx, config, run_name):
    """Creates the requested tracker on the main process; any problem degrades to local logging only."""
    if not ctx.is_main or args.tracker == "none":
        return Tracker()
    if args.tracker == "wandb" and args.wandb_mode == "disabled":
        print("Tracker: wandb is disabled (--wandb-mode disabled); logging locally only.")
        return Tracker()
    try:
        tracker = (WandbTracker if args.tracker == "wandb" else MlflowTracker)(args, config, run_name)
        print(f"Tracker: {args.tracker} (run '{run_name}')")
        return tracker
    except ImportError:
        print(f"WARNING: '{args.tracker}' is not installed (pip install {args.tracker}); "
              "continuing with the local CSV/JSON logs only.")
    except Exception as e:
        print(f"WARNING: could not start {args.tracker} ({type(e).__name__}: {e}); "
              "continuing with the local CSV/JSON logs only.")
    return Tracker()


# =============================================================================
# Main workflow
# =============================================================================
def main():
    parser = build_parser()
    if any(m in sys.modules for m in ("ipykernel", "google.colab")):
        # Inside a notebook kernel, sys.argv also holds the kernel's own flags (e.g. "-f kernel.json");
        # ignore those instead of failing. Outside notebooks, unknown flags remain an error.
        args, ignored = parser.parse_known_args()
        if ignored:
            print(f"(notebook kernel detected; ignoring arguments: {' '.join(ignored)})")
    else:
        args = parser.parse_args()
    determinism = set_seed(args.seed, args.deterministic)
    ctx = init_distributed(args.device)
    ctx.log(json.dumps(vars(args), indent=2))
    device = ctx.device

    # ---- run description --------------------------------------------------
    run_name = args.run_name or os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    if ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    ctx.log(f"[device] {device}" + (f" ({gpu_name})" if gpu_name else "") +
            f" | world size {ctx.world_size}" + (f" ({ctx.backend})" if ctx.enabled else " (single process)") +
            f" | global batch {args.batch_size * ctx.world_size}")
    if device.type == "cpu":
        ctx.log("[device] cpu: training will be slow; use a GPU runtime for realistic speeds.")
    elif args.batch_size < 32:
        ctx.log(f"[device] note: --batch-size {args.batch_size} is small for a GPU; try 64-512.")

    # ---- data -------------------------------------------------------------
    ensure_dataset(args.data, ctx)
    train_loader, val_loader, test_loader, train_sampler, test_set, sizes = build_data(args, ctx)
    ctx.log(f"samples: train {sizes['train']} | validation {sizes['val']} | test {sizes['test']}   "
            f"(augmentation {'on' if args.augment else 'off'})")
    if ctx.is_main:                                            # quick sanity check of what the model will receive
        images, labels = next(iter(train_loader))
        ctx.log(f"First training batch: images {tuple(images.shape)} {images.dtype} "
                f"(min {images.min():.2f}, max {images.max():.2f}), labels {tuple(labels.shape)} {labels.dtype}")

    # ---- model, loss, optimiser -------------------------------------------
    set_seed(args.seed, args.deterministic)                    # same initial weights on every run (and every process)
    model = SimpleCNN().to(device)                             # model lives on the chosen device from here on
    n_params, summary = model_summary(model)
    ctx.log(f"Model: SimpleCNN, {n_params:,} trainable parameters\n{summary}")
    if ctx.enabled:                                            # DDP averages gradients across processes after backward()
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    base_model = model.module if ctx.enabled else model        # unwrapped module for evaluation / checkpoints

    # ---- bookkeeping and tracking (main process only) ----------------------
    config = {**vars(args), "run_name": run_name, "device": str(device), "gpu": gpu_name,
              "world_size": ctx.world_size, "global_batch_size": args.batch_size * ctx.world_size,
              "n_params": n_params, "torch": torch.__version__, "python": sys.version.split()[0],
              "determinism": determinism}
    if ctx.is_main:
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))
        print("\n".join(write_requirements(run_dir / "requirements.txt")))
    tracker = make_tracker(args, ctx, config, run_name)
    metrics_file = None
    if ctx.is_main:
        metrics_file = open(run_dir / "metrics.csv", "w", newline="")
        metrics_writer = csv.writer(metrics_file, lineterminator="\n")
        metrics_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds", "best"])

    # ---- training loop with per-epoch validation ---------------------------
    best_acc, best_loss, best_epoch, step = -1.0, float("inf"), 0, 0
    started = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)                 # new shuffle each epoch, identical on every process
            train_loss, train_acc, step = train_one_epoch(model, train_loader, optimizer, criterion,
                                                          ctx, epoch, args, tracker, step)
            sync_buffers(base_model, ctx)
            val = evaluate(base_model, val_loader, criterion, ctx)

            is_best = val["acc"] > best_acc or (val["acc"] == best_acc and val["loss"] < best_loss)
            if is_best:
                best_acc, best_loss, best_epoch = val["acc"], val["loss"], epoch
                if ctx.is_main:
                    save_state_dict(model, run_dir / "best_model.pth")   # keep the best model by validation accuracy
            seconds = time.time() - epoch_start
            # The launcher's progress indicator keys on the "--- Epoch N:" prefix of this line.
            ctx.log(f"--- Epoch {epoch}: train loss {train_loss:.3f}, train acc {train_acc:.2f}% | "
                    f"val loss {val['loss']:.3f}, val acc {val['acc']:.2f}% | {seconds:.1f}s"
                    f"{' | new best' if is_best else ''} ---")
            if ctx.is_main:
                metrics_writer.writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.3f}", f"{val['loss']:.5f}",
                                         f"{val['acc']:.3f}", f"{seconds:.1f}", int(is_best)])
                metrics_file.flush()
                save_state_dict(model, run_dir / "last_model.pth")
            tracker.log_metrics({"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc,
                                 "val/loss": val["loss"], "val/acc": val["acc"],
                                 "epoch_seconds": seconds}, step)
        ctx.log(f"Finished training in {time.time() - started:.1f}s; best validation accuracy "
                f"{best_acc:.2f}% at epoch {best_epoch}")

        # ---- final evaluation of the saved best model ----------------------
        ctx.barrier()                                          # make sure best_model.pth is fully written before anyone reads it
        best_model = SimpleCNN().to(device)
        best_model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
        test = evaluate(best_model, test_loader, criterion, ctx)
        report = per_class_report(test["confusion"])
        macro = {m: float(np.mean([report[c][m] for c in CLASSES])) for m in ("precision", "recall", "f1")}
        ctx.log(f"Test accuracy of the best model: {test['acc']:.2f}% (loss {test['loss']:.3f}) | "
                f"macro precision {macro['precision']:.3f}, recall {macro['recall']:.3f}, F1 {macro['f1']:.3f}")

        if ctx.is_main:
            print(f"{'class':<8}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}")
            for c in CLASSES:
                r = report[c]
                print(f"{c:<8}{r['precision']:>10.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}{r['support']:>9}")
            print("Confusion matrix (rows = true class, columns = predicted):")
            print("        " + "".join(f"{c[:5]:>6}" for c in CLASSES))
            for i, c in enumerate(CLASSES):
                print(f"{c:<8}" + "".join(f"{v:>6}" for v in test["confusion"][i]))
            csv_path, png_path = save_confusion_matrix(test["confusion"], run_dir)
            final = {"test_loss": test["loss"], "test_acc": test["acc"], "macro": macro,
                     "per_class": report, "best_epoch": best_epoch, "best_val_acc": best_acc,
                     "train_seconds": time.time() - started, "world_size": ctx.world_size}
            (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
            inference_demo(best_model, test_set, device)
            tracker.log_summary({"final/test_acc": test["acc"], "final/test_loss": test["loss"],
                                 "final/macro_f1": macro["f1"], "final/best_val_acc": best_acc,
                                 "final/best_epoch": best_epoch})
            if png_path:
                tracker.log_image("test/confusion_matrix", png_path)
            tracker.log_outputs([run_dir / "best_model.pth"],
                                [p for p in (csv_path, png_path, run_dir / "metrics.csv", run_dir / "run_config.json",
                                             run_dir / "final_metrics.json", run_dir / "requirements.txt") if p])
            print(f"Results written to {run_dir}")
    except torch.cuda.OutOfMemoryError:
        ctx.log("ERROR: CUDA out of memory. Lower --batch-size (it is per GPU), reduce --num-workers, "
                "or free the GPU (check nvidia-smi for other processes).")
        raise SystemExit(1)
    finally:                                                   # always release files and finish the tracker run
        if metrics_file:
            metrics_file.close()
        tracker.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)                                          # conventional exit status for Ctrl-C
    finally:                                                   # leave the process group cleanly, whatever happened
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
