import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(histories: dict, title: str = "Training curves"):
    """
    histories: {exp_name: {"train_loss": [...], "train_acc": [...], "val_loss": [...], "val_acc": [...]}}
    Solid lines = train, dashed = val.
    """
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    for i, (name, h) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        epochs = range(1, len(h['train_loss']) + 1)
        axes[0].plot(epochs, h['train_loss'], color=c, linestyle='-',  label=f"{name} train")
        axes[0].plot(epochs, h['val_loss'],   color=c, linestyle='--', label=f"{name} val")
        axes[1].plot(epochs, h['train_acc'],  color=c, linestyle='-',  label=f"{name} train")
        axes[1].plot(epochs, h['val_acc'],    color=c, linestyle='--', label=f"{name} val")

    for ax, ylabel, title_ in zip(axes, ['Loss', 'Accuracy'], ['Loss', 'Accuracy']):
        ax.set_title(title_)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_class_distribution(glosses_path: Path) -> pd.DataFrame:
    """
    Prints a table and stacked bar chart of instances per gloss per split.
    Returns the DataFrame with columns [train, val, test, total].
    """
    content = json.loads(Path(glosses_path).read_text())

    rows = [
        {"gloss": entry["gloss"], "split": inst["split"]}
        for entry in content
        for inst in entry["instances"]
    ]
    df = pd.DataFrame(rows)

    table = df.groupby(["gloss", "split"]).size().unstack(fill_value=0)
    for col in ["train", "val", "test"]:
        if col not in table.columns:
            table[col] = 0
    table = table[["train", "val", "test"]]
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False)

    # ── Tabla ─────────────────────────────────────────────────────────────────
    print(f"{'Gloss':<35} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-" * 62)
    for gloss, row in table.iterrows():
        print(f"{gloss:<35} {row['train']:>6} {row['val']:>6} {row['test']:>6} {row['total']:>7}")
    print("-" * 62)
    totals = table[["train", "val", "test", "total"]].sum()
    print(f"{'TOTAL':<35} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6} {totals['total']:>7}")

    # ── Gráfico ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(table) * 0.35), 5))
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    bottom = [0] * len(table)
    x = range(len(table))
    for split in ["train", "val", "test"]:
        ax.bar(x, table[split], bottom=bottom, label=split, color=colors[split])
        bottom = [b + v for b, v in zip(bottom, table[split])]

    ax.set_xticks(list(x))
    ax.set_xticklabels(table.index, rotation=90, fontsize=7)
    ax.set_xlabel("Gloss")
    ax.set_ylabel("Muestras")
    ax.set_title("Distribución de muestras por clase y split")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return table


def show_augmentation_example(nslt_aug_json: Path, videos_dir: Path, n_frames: int = 6, seed: int = 42, aug_seed: int = 42):
    """
    Picks a random training video that has both augmented versions and displays
    a grid of sampled frames: Original / Temporal rescale / Filtro visual.
    """
    import random
    import cv2
    import numpy as np

    random.seed(seed)
    nslt = json.loads(Path(nslt_aug_json).read_text())
    videos_dir = Path(videos_dir)

    candidates = [
        vid_id for vid_id, info in nslt.items()
        if info["subset"] == "train"
        and not vid_id.endswith("t")
        and not vid_id.endswith("f")
        and (videos_dir / f"{vid_id}.mp4").exists()
        and (videos_dir / f"{vid_id}t.mp4").exists()
        and (videos_dir / f"{vid_id}f.mp4").exists()
    ]

    if not candidates:
        print("No se encontraron videos con ambas versiones augmentadas.")
        return

    vid_id = random.choice(candidates)

    # Reconstruct augmentation params deterministically
    import hashlib
    digest = hashlib.sha256(f"{aug_seed}:{vid_id}".encode()).digest()
    video_seed = int.from_bytes(digest[:4], "big")
    _rng = random.Random(video_seed)
    scale = _rng.uniform(0.5, 2.0)
    filter_names = [
        "RandomBrightnessContrast (fuerte)",
        "HueSaturationValue",
        "GaussNoise",
        "GaussianBlur",
        "CLAHE",
        "ImageCompression",
        "RandomBrightnessContrast (suave) + GaussNoise",
        "HueSaturationValue + GaussianBlur (suave)",
    ]
    filter_idx = _rng.randrange(len(filter_names))
    speed = "lento" if scale > 1 else "rápido"
    print(f"Video de ejemplo  : {vid_id}")
    print(f"Temporal rescale  : scale={scale:.3f} ({speed})")
    print(f"Filtro visual     : {filter_names[filter_idx]}")

    versions = [
        ("Original",          videos_dir / f"{vid_id}.mp4"),
        ("Temporal rescale",  videos_dir / f"{vid_id}t.mp4"),
        ("Filtro visual",     videos_dir / f"{vid_id}f.mp4"),
    ]

    def sample_frames(path, n):
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [int(i * (total - 1) / (n - 1)) for i in range(n)] if total >= n else list(range(total))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 2.5, 7))
    fig.suptitle(f"Augmentation — video: {vid_id}", fontsize=13)

    for row, (label, path) in enumerate(versions):
        frames = sample_frames(path, n_frames)
        for col in range(n_frames):
            ax = axes[row][col]
            if col < len(frames):
                ax.imshow(frames[col])
            else:
                ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=90, va="center")

    plt.tight_layout()
    plt.show()
