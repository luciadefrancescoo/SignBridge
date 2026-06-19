"""
Offline data augmentation for WLASL training videos.

Generates two augmented copies per training video:
  - <vid>t.mp4 : temporal rescale (random speed, fast or slow)
  - <vid>f.mp4 : albumentations visual filter (randomly chosen per video)

Both copies are added to a new JSON split file so the NSLT dataset loader
picks them up automatically. Augmented videos are saved to --out-dir (defaults
to the same directory as the source videos so no dataset code changes needed).

Usage:
    # All classes (WLASL2000)
    python augment_offline.py \
        --json     preprocess/nslt_2000.json \
        --vid-root /path/to/raw_videos \
        --out-json preprocess/nslt_2000_aug.json \
        --workers  24 \
        --seed 42
    

    # Only WLASL100 videos (filter with the WLASL100 metadata JSON)
    python augment_offline.py \\
        --json        preprocess/nslt_2000.json \\
        --filter-json start_kit/WLASL100.json \\
        --vid-root    /path/to/raw_videos \\
        --out-json    preprocess/nslt_100_aug.json \\
        --workers     24 \\
        --seed        42
"""

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import albumentations as A
import cv2
import numpy as np
from scipy.ndimage import zoom as ndimage_zoom


def derive_seed(seed: int, vid: str) -> int:
    """
    Deterministic per-video seed derived from the global --seed and video id.
    Needed because videos are processed across worker processes in
    non-deterministic order (as_completed) - keying off the video id makes
    each video's augmentation reproducible regardless of run-to-run
    scheduling or worker count.
    """
    digest = hashlib.sha256(f"{seed}:{vid}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# Temporal augmentation
# ---------------------------------------------------------------------------

def temporal_rescale(frames: np.ndarray, scale: float) -> np.ndarray:
    """
    Rescale the temporal axis of `frames` by `scale`, then restore the original
    frame count.
      scale > 1 → slow motion: interpolate extra frames, keep first T
      scale < 1 → fast motion: compress frames, pad end with last frame
    Args:
        frames: uint8 array (T, H, W, C)
        scale:  positive float
    Returns:
        uint8 array (T, H, W, C)
    """
    t = frames.shape[0]
    new_t = max(1, int(round(t * scale)))

    work = frames.astype(np.float32)
    rescaled = ndimage_zoom(work, (new_t / t, 1.0, 1.0, 1.0), order=1)

    curr_t = rescaled.shape[0]
    if curr_t >= t:
        result = rescaled[:t]
    else:
        pad = np.repeat(rescaled[[-1]], t - curr_t, axis=0)
        result = np.concatenate([rescaled, pad], axis=0)

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visual filter augmentation
# ---------------------------------------------------------------------------

# Each pipeline is wrapped in ReplayCompose so the same random parameters are
# applied to every frame (avoids per-frame flickering). The pipeline itself is
# chosen randomly per video, giving variability across the dataset.
FILTER_PIPELINES = [
    A.ReplayCompose([
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
    ]),
    A.ReplayCompose([
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=25, p=1.0),
    ]),
    A.ReplayCompose([
        A.GaussNoise(var_limit=(15, 60), p=1.0),
    ]),
    A.ReplayCompose([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ]),
    A.ReplayCompose([
        A.CLAHE(clip_limit=(2.0, 6.0), tile_grid_size=(8, 8), p=1.0),
    ]),
    A.ReplayCompose([
        A.ImageCompression(quality_lower=40, quality_upper=75, p=1.0),
    ]),
    A.ReplayCompose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.GaussNoise(var_limit=(10, 30), p=0.7),
    ]),
    A.ReplayCompose([
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ]),
]


def apply_filter(frames: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply a randomly chosen visual filter consistently across all frames.
    Args:
        frames: uint8 array (T, H, W, C) in BGR
        seed:   if given, seeds the chosen pipeline's RNG for reproducibility
                (albumentations keeps its own generator, separate from the
                `random` module, so seeding that module alone isn't enough)
    Returns:
        uint8 array (T, H, W, C)
    """
    pipeline = random.choice(FILTER_PIPELINES)
    if seed is not None:
        pipeline.set_random_seed(seed)

    # Apply to first frame to capture replay parameters
    first_result = pipeline(image=frames[0])
    replay = first_result["replay"]

    result = [first_result["image"]]
    for frame in frames[1:]:
        result.append(A.ReplayCompose.replay(replay, image=frame)["image"])

    return np.stack(result, axis=0)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def read_video(path: str, max_side: int = 256):
    """
    Returns (frames uint8 (T,H,W,C), fps float).
    Frames are downscaled so the longer side is at most `max_side` pixels,
    matching the I3D preprocessing limit and reducing temporal zoom cost.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        return np.empty((0,), dtype=np.uint8), fps
    return np.stack(frames, axis=0), fps


def write_video(frames: np.ndarray, path: str, fps: float) -> None:
    if frames.shape[0] == 0:
        return
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Worker (runs in subprocess)
# ---------------------------------------------------------------------------

def process_video(task: dict) -> dict:
    """
    Augment one video and return a dict with the new JSON entries.
    Designed to run inside ProcessPoolExecutor workers.
    """
    vid = task["vid"]
    entry = task["entry"]
    vid_root = task["vid_root"]
    out_dir = task["out_dir"]
    scale_range = task["scale_range"]
    seed = task["seed"]

    if seed is not None:
        video_seed = derive_seed(seed, vid)
        random.seed(video_seed)
    else:
        video_seed = None

    src_path = os.path.join(vid_root, vid + ".mp4")
    if not os.path.exists(src_path):
        return {"ok": False, "vid": vid, "error": "not_found"}

    frames, fps = read_video(src_path)
    if frames.shape[0] == 0:
        return {"ok": False, "vid": vid, "error": "unreadable"}

    class_id = entry["action"][0]
    n_frames = int(frames.shape[0])
    new_action = [class_id, 0, n_frames]

    # Temporal augmentation (random scale: fast or slow)
    scale = random.uniform(*scale_range)
    t_frames = temporal_rescale(frames, scale)
    t_vid_id = vid + "t"
    write_video(t_frames, os.path.join(out_dir, t_vid_id + ".mp4"), fps)

    # Visual filter augmentation
    f_frames = apply_filter(frames, seed=video_seed)
    f_vid_id = vid + "f"
    write_video(f_frames, os.path.join(out_dir, f_vid_id + ".mp4"), fps)

    return {
        "ok": True,
        "vid": vid,
        "t_vid_id": t_vid_id,
        "t_entry": {"subset": "train", "action": new_action},
        "f_vid_id": f_vid_id,
        "f_entry": {"subset": "train", "action": new_action},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline video augmentation for WLASL")
    parser.add_argument("--json", required=True, help="Source nslt split JSON (e.g. nslt_2000.json)")
    parser.add_argument(
        "--filter-json",
        default=None,
        help="WLASL metadata JSON (WLASL100.json / WLASL300.json). "
             "When provided, only video IDs present in that file are processed.",
    )
    parser.add_argument("--vid-root", required=True, help="Directory containing source MP4s")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save augmented MP4s (default: same as --vid-root)",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON with augmented entries")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() - 4),
        help="Parallel worker processes (default: cpu_count - 4)",
    )
    parser.add_argument("--scale-min", type=float, default=0.5, help="Min temporal scale")
    parser.add_argument("--scale-max", type=float, default=2.0, help="Max temporal scale")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible augmentation (default: none, non-deterministic)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.vid_root
    os.makedirs(out_dir, exist_ok=True)

    with open(args.json) as f:
        data = json.load(f)

    # Optional: restrict to the video IDs present in a WLASL subset metadata file
    if args.filter_json:
        with open(args.filter_json) as f:
            wlasl_meta = json.load(f)
        allowed_ids = {inst["video_id"] for entry in wlasl_meta for inst in entry["instances"]}
        data = {vid: entry for vid, entry in data.items() if vid in allowed_ids}
        print(f"Filtered to {len(data)} videos from {args.filter_json}")

    train_vids = [(vid, entry) for vid, entry in data.items() if entry["subset"] == "train"]
    print(f"Training videos found : {len(train_vids)}")
    print(f"Workers               : {args.workers}")
    print(f"Temporal scale range  : [{args.scale_min}, {args.scale_max}]")
    print(f"Output directory      : {out_dir}")
    print(f"Output JSON           : {args.out_json}")
    print(f"Seed                  : {args.seed}\n")

    tasks = [
        {
            "vid": vid,
            "entry": entry,
            "vid_root": args.vid_root,
            "out_dir": out_dir,
            "scale_range": (args.scale_min, args.scale_max),
            "seed": args.seed,
        }
        for vid, entry in train_vids
    ]

    new_data = dict(data)
    done = 0
    failed = 0
    failed_by_reason = {"not_found": 0, "unreadable": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_video, t): t["vid"] for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            done += 1
            if res["ok"]:
                new_data[res["t_vid_id"]] = res["t_entry"]
                new_data[res["f_vid_id"]] = res["f_entry"]
            else:
                failed += 1
                failed_by_reason[res["error"]] += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} processed  |  failed: {failed}")

    with open(args.out_json, "w") as f:
        json.dump(new_data, f, indent=2)

    added = len(new_data) - len(data)
    print(f"\nDone.")
    print(f"  Original entries  : {len(data)}")
    print(f"  Added entries     : {added}")
    print(f"  Total entries     : {len(new_data)}")
    print(f"  Failed videos     : {failed}")
    print(f"    not found       : {failed_by_reason['not_found']}  (missing from --vid-root)")
    print(f"    unreadable      : {failed_by_reason['unreadable']}  (corrupt file / 0 decodable frames)")
    print(f"  Saved to          : {args.out_json}")


if __name__ == "__main__":
    main()
