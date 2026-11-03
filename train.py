#!/usr/bin/env python3
"""CIFAR-10 CNN training (script version of the notebook, GPU-enabled).

Run directly:        python train.py --epochs 10 --seed 7
Run via the wrapper: bash run_distributed.sh --epochs 10 --seed 7

Device selection (--device auto, the default):
  1. $RUN_DEVICE, if set (the wrapper exports "cuda" or "cpu" after probing the hardware)
  2. otherwise CUDA if PyTorch can use it, else CPU
"""
import argparse
import glob
import io
import json
import os
import random
import sys
from importlib import metadata as md

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CIFAR-10 CNN training")

g = parser.add_argument_group("reproducibility")
g.add_argument("--seed", type=int, default=42)
g.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
               help="deterministic algorithms / cuDNN settings (use --no-deterministic to disable)")

g = parser.add_argument_group("training")
g.add_argument("--epochs", type=int, default=4)
g.add_argument("--batch-size", type=int, default=4)
g.add_argument("--lr", type=float, default=0.001)
g.add_argument("--momentum", type=float, default=0.9)
g.add_argument("--num-workers", type=int, default=2)
g.add_argument("--log-interval", type=int, default=2000, help="print/log every N mini-batches")
g.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
               help="auto = $RUN_DEVICE if set, else CUDA when available, else CPU")

g = parser.add_argument_group("paths")
g.add_argument("--data-dir", default="/kaggle/working", help="where the HF dataset is downloaded/read")
g.add_argument("--save-path", default="./cifar_net.pth")

g = parser.add_argument_group("experiment tracking (Weights & Biases)")
g.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
g.add_argument("--wandb-project", default="cifar10-cnn")
g.add_argument("--wandb-run-name", default=None)

# parse_known_args so Jupyter's own flags (-f kernel.json) never break a notebook run
args, _unknown = parser.parse_known_args()
print(json.dumps(vars(args), indent=2))


# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------
def pick_device(requested):
    """Resolve --device / $RUN_DEVICE into a torch.device, failing loudly if CUDA is demanded but absent."""
    if requested == "auto":
        requested = os.environ.get("RUN_DEVICE", "").strip().lower() or "auto"
        if requested not in ("auto", "cuda", "cpu"):
            print(f"[device] ignoring unrecognised RUN_DEVICE={requested!r}")
            requested = "auto"
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        sys.exit("ERROR: CUDA was requested but torch.cuda.is_available() is False "
                 "(no GPU, CPU-only PyTorch build, or driver mismatch). Use --device cpu to run on the CPU.")
    return torch.device("cpu")


device = pick_device(args.device)
if device.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"[device] cuda:0 {p.name} ({p.total_memory // 2**20} MiB), "
          f"{torch.cuda.device_count()} visible GPU(s); training on 1 GPU")
else:
    print("[device] cpu (training will be slow; see --device / the wrapper's GPU diagnostics)")
if device.type == "cuda" and args.batch_size < 32:
    print(f"[device] note: --batch-size {args.batch_size} is tiny for a GPU; "
          "try 64-512 to actually benefit from it.")


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed, deterministic=True):
    """Seed every RNG we use and (optionally) force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)                  # only affects subprocesses
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)                          # no-op without CUDA
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"     # needed for deterministic cuBLAS on GPU
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """Gives each DataLoader worker a distinct but reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(args.seed, args.deterministic)
train_generator = torch.Generator().manual_seed(args.seed)    # controls the shuffle order of trainloader


def write_requirements(path="requirements.txt",
                       packages=("torch", "torchvision", "numpy", "pandas", "pyarrow", "pillow",
                                 "matplotlib", "huggingface_hub", "wandb")):
    """Pins the versions installed in THIS environment so the run can be recreated."""
    lines = [f"# python {sys.version.split()[0]}  |  cuda {torch.version.cuda}"]
    for pkg in packages:
        try:
            lines.append(f"{pkg}=={md.version(pkg)}")
        except md.PackageNotFoundError:
            lines.append(f"# {pkg}: not installed")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


print("\n".join(write_requirements()))

# ----------------------------------------------------------------------------
# Weights & Biases (optional)
# ----------------------------------------------------------------------------
try:
    import wandb
except ImportError:
    wandb = None
    if args.wandb_mode != "disabled":
        print("wandb is not installed -> tracking disabled (pip install wandb to enable).")

run = None


def wb_log(data, step=None):
    """No-op when tracking is off, so the training code never needs to check."""
    if run is not None:
        run.log(data, step=step)


if wandb is not None and args.wandb_mode != "disabled":
    # API key: WANDB_API_KEY env var, else a Kaggle secret named WANDB_API_KEY
    if "WANDB_API_KEY" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception as e:
            print(f"No WANDB_API_KEY found ({type(e).__name__}).")

    config = {**vars(args),
              "python": sys.version.split()[0], "torch": torch.__version__,
              "device": str(device),
              "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "cuda_available": torch.cuda.is_available(),
              "dataset": "CIFAR10 (HF: uoft-cs/cifar10)"}
    try:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                         config=config, mode=args.wandb_mode)
    except Exception as e:                                    # e.g. no internet or bad key
        print(f"W&B init failed ({type(e).__name__}: {e}); falling back to offline mode.")
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                         config=config, mode="offline")
    run.save("requirements.txt")

# ----------------------------------------------------------------------------
# Data (CIFAR-10 from the Hugging Face hub, which was much faster than torchvision's mirror)
# ----------------------------------------------------------------------------
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def find_parquet(split):
    files = sorted(glob.glob(f"{args.data_dir}/**/{split}-*.parquet", recursive=True))
    return files[0] if files else None


if find_parquet("train") is None or find_parquet("test") is None:
    from huggingface_hub import snapshot_download
    print(f"Downloading CIFAR-10 to {args.data_dir} ...")
    snapshot_download(repo_id="uoft-cs/cifar10", repo_type="dataset", cache_dir=args.data_dir)
else:
    print(f"CIFAR-10 already present under {args.data_dir}; skipping download.")

for _split in ("train", "test"):
    assert find_parquet(_split), f"No {_split} parquet found under {args.data_dir}"


class HFCIFAR10(Dataset):
    """Drop-in replacement for torchvision.datasets.CIFAR10 using HF parquet files."""

    def __init__(self, parquet_path, transform=None):
        df = pd.read_parquet(parquet_path)
        self.images = df["img"].tolist()                      # dicts with 'bytes' (PNG data)
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]["bytes"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# pin_memory speeds up host->GPU copies; it only makes sense (and only works) with CUDA
loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     worker_init_fn=seed_worker, pin_memory=(device.type == "cuda"))

trainset = HFCIFAR10(find_parquet("train"), transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, generator=train_generator, **loader_kwargs)

testset = HFCIFAR10(find_parquet("test"), transform=transform)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, **loader_kwargs)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

print(f"train/test sizes: {len(trainset)} / {len(testset)}   (expect 50000 / 10000)")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                               # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


set_seed(args.seed, args.deterministic)                       # identical initial weights every run
net = Net().to(device)                                        # <-- model lives on the chosen device
n_params = sum(p.numel() for p in net.parameters())
print(f"Net: {n_params:,} trainable parameters on {device}")
if run is not None:
    run.config.update({"n_params": n_params})

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)   # created AFTER .to(device)


def to_device(batch):
    """Moves a (inputs, labels) batch to the model's device (async when the loader pinned memory)."""
    inputs, labels = batch
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def evaluate(model, loader):
    """Returns (avg loss, accuracy %) over a full dataloader, computed on `device`."""
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = to_device(batch)
            outputs = model(inputs)
            # criterion returns a batch mean, so weight by batch size (last batch may be smaller)
            total_loss += criterion(outputs, labels) * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum()
            total += labels.size(0)
    model.train(was_training)
    return total_loss.item() / total, 100 * correct.item() / total


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
# Re-seed the shuffling RNG so every run sees the same batch order
train_generator.manual_seed(args.seed)
global_step = 0
net.train()

for epoch in range(args.epochs):
    # Statistics are accumulated as tensors ON the device and only read with .item() when
    # logging. Calling loss.item() every step would force a GPU->CPU sync per mini-batch.
    run_loss = torch.zeros((), device=device)
    run_correct = torch.zeros((), device=device, dtype=torch.long)
    run_total = 0
    ep_loss = torch.zeros((), device=device)
    ep_correct = torch.zeros((), device=device, dtype=torch.long)
    ep_total = 0

    for i, batch in enumerate(trainloader):
        inputs, labels = to_device(batch)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        global_step += 1

        n = labels.size(0)
        batch_correct = (outputs.argmax(dim=1) == labels).sum()
        run_loss += loss.detach()
        run_correct += batch_correct
        run_total += n
        ep_loss += loss.detach() * n
        ep_correct += batch_correct
        ep_total += n

        if i % args.log_interval == args.log_interval - 1:    # print/log every --log-interval mini-batches
            avg_loss = run_loss.item() / args.log_interval
            avg_acc = 100 * run_correct.item() / run_total
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}  acc: {avg_acc:.2f}%")
            wb_log({"train/loss": avg_loss, "train/acc": avg_acc, "epoch": epoch + 1}, step=global_step)
            run_loss.zero_()
            run_correct.zero_()
            run_total = 0

    # end-of-epoch summary (the wrapper's progress indicator keys on this exact "--- Epoch N:" format)
    train_loss = ep_loss.item() / ep_total
    train_acc = 100 * ep_correct.item() / ep_total
    test_loss, test_acc = evaluate(net, testloader)
    print(f"--- Epoch {epoch + 1}: "
          f"train loss {train_loss:.3f}, train acc {train_acc:.2f}% | "
          f"test loss {test_loss:.3f}, test acc {test_acc:.2f}% ---")
    wb_log({"epoch": epoch + 1,
            "epoch/train_loss": train_loss, "epoch/train_acc": train_acc,
            "epoch/test_loss": test_loss, "epoch/test_acc": test_acc}, step=global_step)

print("Finished Training")

# ----------------------------------------------------------------------------
# Save, reload and final evaluation
# ----------------------------------------------------------------------------
PATH = args.save_path
os.makedirs(os.path.dirname(os.path.abspath(PATH)), exist_ok=True)
# Save CPU tensors so the checkpoint loads on any machine, with or without a GPU
torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, PATH)
print(f"Saved weights to {PATH}")

if run is not None:
    artifact = wandb.Artifact("cifar-net", type="model",
                              metadata={"seed": args.seed, "epochs": args.epochs, "device": str(device)})
    artifact.add_file(PATH)
    artifact.add_file("requirements.txt")
    run.log_artifact(artifact)

net = Net().to(device)
net.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))
net.eval()

# A few sample predictions
images, labels = next(iter(testloader))
images, labels = images[:4], labels[:4]
with torch.no_grad():
    predicted = net(images.to(device)).argmax(dim=1).cpu()
print("GroundTruth:", " ".join(f"{classes[labels[j]]:5s}" for j in range(len(labels))))
print("Predicted:  ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(len(predicted))))

# Same code path as the end-of-epoch test metric, so the numbers are directly comparable
test_loss, test_acc = evaluate(net, testloader)
print(f"Accuracy of the network on the {len(testset)} test images: {test_acc:.2f} %  (test loss {test_loss:.3f})")
if run is not None:
    run.summary["final/test_acc"] = test_acc
    run.summary["final/test_loss"] = test_loss

# Per-class accuracy
correct_pred = {c: 0 for c in classes}
total_pred = {c: 0 for c in classes}
with torch.no_grad():
    for batch in testloader:
        inputs, labels = to_device(batch)
        preds = net(inputs).argmax(dim=1)
        for label, prediction in zip(labels.tolist(), preds.tolist()):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

per_class = {}
for classname in classes:
    per_class[classname] = 100 * correct_pred[classname] / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {per_class[classname]:.1f} %")

if run is not None:
    for c, acc in per_class.items():
        run.summary[f"final/class_acc/{c}"] = acc
    table = wandb.Table(data=[[c, a] for c, a in per_class.items()], columns=["class", "accuracy"])
    run.log({"final/per_class_accuracy": wandb.plot.bar(table, "class", "accuracy",
                                                        title="Per-class test accuracy (%)")})
    run.finish()   # in offline mode, upload later with: wandb sync wandb/offline-run-*
