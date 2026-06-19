#!/usr/bin/env python3
"""
Add global hand_bbox metadata to a WLASL-style JSON.

Input WLASL instance format:
[
  {
    "gloss": "book",
    "instances": [
      {
        "bbox": [xmin, ymin, xmax, ymax],
        "video_id": "69241",
        ...
      }
    ]
  }
]

Expected OpenPose folder format:
pose_root/
  69241/
    image_00001_keypoints.json
    image_00002_keypoints.json
    ...

Output:
The original JSON structure is preserved. Each instance receives:
  - hand_bbox: [xmin, ymin, xmax, ymax]
  - hand_bbox_valid: bool
  - hand_bbox_num_valid_points: int
  - hand_bbox_num_frames_used: int
  - hand_bbox_source: "openpose_hands" or "fallback_original_bbox"

uv run python add_hand_bbox_to_wlasl_json.py \
  --input-json WLASL_v0.3.full.json \
  --pose-root WLASL/TGCN/data/pose_per_individual_videos \
  --output-json WLASL_v0.3.full_hand_bbox.json

By default, if no valid hand keypoints are found, the script uses the original
instance["bbox"] as fallback and marks hand_bbox_valid = false.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


BBox = List[int]
Point = Tuple[float, float]


def parse_flat_keypoints(
    flat_keypoints: Sequence[float],
    conf_threshold: float,
) -> List[Point]:
    """
    Convert OpenPose flat keypoints [x1, y1, c1, x2, y2, c2, ...]
    into valid points [(x, y), ...].

    Points with x=0, y=0 or confidence below threshold are ignored.
    """
    points: List[Point] = []

    if not flat_keypoints:
        return points

    for i in range(0, len(flat_keypoints), 3):
        if i + 2 >= len(flat_keypoints):
            break

        x = float(flat_keypoints[i])
        y = float(flat_keypoints[i + 1])
        c = float(flat_keypoints[i + 2])

        if c >= conf_threshold and x > 0 and y > 0:
            points.append((x, y))

    return points


def bbox_from_points(points: Sequence[Point], margin: float) -> BBox:
    """
    Build [xmin, ymin, xmax, ymax] from points and expand it by margin.

    margin = 0.25 means adding 25% of bbox width/height on each side.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)

    xmin -= margin * width
    ymin -= margin * height
    xmax += margin * width
    ymax += margin * height

    return [
        int(round(xmin)),
        int(round(ymin)),
        int(round(xmax)),
        int(round(ymax)),
    ]


def clip_bbox(inner_bbox: BBox, outer_bbox: BBox) -> BBox:
    """
    Clip inner_bbox so that it stays inside outer_bbox.
    Both bboxes use [xmin, ymin, xmax, ymax].
    """
    xmin, ymin, xmax, ymax = inner_bbox
    oxmin, oymin, oxmax, oymax = outer_bbox

    xmin = max(xmin, oxmin)
    ymin = max(ymin, oymin)
    xmax = min(xmax, oxmax)
    ymax = min(ymax, oymax)

    # If clipping created an invalid bbox, return the outer bbox as safe fallback.
    if xmax <= xmin or ymax <= ymin:
        return [int(oxmin), int(oymin), int(oxmax), int(oymax)]

    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def get_keypoint_files(video_pose_dir: Path) -> List[Path]:
    """
    Return sorted OpenPose keypoint JSON files for one video.
    """
    patterns = [
        "*_keypoints.json",
        "*.json",
    ]

    files: List[Path] = []
    for pattern in patterns:
        files = sorted(video_pose_dir.glob(pattern))
        if files:
            break

    return files


def load_first_person(keypoint_file: Path) -> Optional[Dict]:
    """
    Load one OpenPose JSON and return the first person if available.
    """
    try:
        with keypoint_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    people = data.get("people", [])
    if not people:
        return None

    return people[0]


def compute_global_hand_bbox(
    video_id: str,
    pose_root: Path,
    original_bbox: BBox,
    conf_threshold: float,
    margin: float,
    min_valid_points: int,
    clip_to_original_bbox: bool,
) -> Dict:
    """
    Compute one global hand bbox for a video using all available OpenPose frames.
    """
    video_pose_dir = pose_root / str(video_id)

    if not video_pose_dir.exists():
        return {
            "hand_bbox": original_bbox,
            "hand_bbox_valid": False,
            "hand_bbox_num_valid_points": 0,
            "hand_bbox_num_frames_used": 0,
            "hand_bbox_source": "fallback_original_bbox",
            "hand_bbox_error": "pose_dir_not_found",
        }

    keypoint_files = get_keypoint_files(video_pose_dir)

    if not keypoint_files:
        return {
            "hand_bbox": original_bbox,
            "hand_bbox_valid": False,
            "hand_bbox_num_valid_points": 0,
            "hand_bbox_num_frames_used": 0,
            "hand_bbox_source": "fallback_original_bbox",
            "hand_bbox_error": "no_keypoint_files",
        }

    all_points: List[Point] = []
    frames_used = 0

    for keypoint_file in keypoint_files:
        person = load_first_person(keypoint_file)
        if person is None:
            continue

        left_hand = person.get("hand_left_keypoints_2d", [])
        right_hand = person.get("hand_right_keypoints_2d", [])

        frame_points = []
        frame_points.extend(parse_flat_keypoints(left_hand, conf_threshold))
        frame_points.extend(parse_flat_keypoints(right_hand, conf_threshold))

        if frame_points:
            frames_used += 1
            all_points.extend(frame_points)

    if len(all_points) < min_valid_points:
        return {
            "hand_bbox": original_bbox,
            "hand_bbox_valid": False,
            "hand_bbox_num_valid_points": len(all_points),
            "hand_bbox_num_frames_used": frames_used,
            "hand_bbox_source": "fallback_original_bbox",
            "hand_bbox_error": "not_enough_valid_hand_points",
        }

    hand_bbox = bbox_from_points(all_points, margin=margin)

    if clip_to_original_bbox:
        hand_bbox = clip_bbox(hand_bbox, original_bbox)

    return {
        "hand_bbox": hand_bbox,
        "hand_bbox_valid": True,
        "hand_bbox_num_valid_points": len(all_points),
        "hand_bbox_num_frames_used": frames_used,
        "hand_bbox_source": "openpose_hands",
        "hand_bbox_error": None,
    }


def add_hand_bboxes_to_dataset(
    dataset: List[Dict],
    pose_root: Path,
    conf_threshold: float,
    margin: float,
    min_valid_points: int,
    clip_to_original_bbox: bool,
) -> Tuple[List[Dict], Dict]:
    """
    Add hand_bbox fields to a WLASL-style dataset.
    """
    output = deepcopy(dataset)

    stats = {
        "num_instances": 0,
        "num_valid_hand_bbox": 0,
        "num_fallback": 0,
        "num_pose_dir_not_found": 0,
        "num_no_keypoint_files": 0,
        "num_not_enough_valid_hand_points": 0,
    }

    for gloss_entry in output:
        for instance in gloss_entry.get("instances", []):
            stats["num_instances"] += 1

            video_id = str(instance["video_id"])
            original_bbox = [int(v) for v in instance["bbox"]]

            result = compute_global_hand_bbox(
                video_id=video_id,
                pose_root=pose_root,
                original_bbox=original_bbox,
                conf_threshold=conf_threshold,
                margin=margin,
                min_valid_points=min_valid_points,
                clip_to_original_bbox=clip_to_original_bbox,
            )

            instance.update(result)

            if result["hand_bbox_valid"]:
                stats["num_valid_hand_bbox"] += 1
            else:
                stats["num_fallback"] += 1
                error = result.get("hand_bbox_error")
                if error == "pose_dir_not_found":
                    stats["num_pose_dir_not_found"] += 1
                elif error == "no_keypoint_files":
                    stats["num_no_keypoint_files"] += 1
                elif error == "not_enough_valid_hand_points":
                    stats["num_not_enough_valid_hand_points"] += 1

    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add global hand_bbox fields to a WLASL-style JSON."
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to original WLASL JSON, e.g. start_kit/WLASL_v0.3.json",
    )
    parser.add_argument(
        "--pose-root",
        required=True,
        help="Root folder with per-video OpenPose JSON folders, e.g. data/pose_per_individual_videos",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path where the enriched JSON will be saved.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.2,
        help="Minimum OpenPose confidence for a hand point to be used. Default: 0.2",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="Margin added to the hand bbox as a fraction of width/height. Default: 0.25",
    )
    parser.add_argument(
        "--min-valid-points",
        type=int,
        default=10,
        help="Minimum valid hand keypoints across the whole video. Default: 10",
    )
    parser.add_argument(
        "--no-clip-to-original-bbox",
        action="store_true",
        help="Do not force hand_bbox to stay inside the original WLASL signer bbox.",
    )

    args = parser.parse_args()

    input_json_path = Path(args.input_json)
    pose_root = Path(args.pose_root)
    output_json_path = Path(args.output_json)

    with input_json_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    enriched_dataset, stats = add_hand_bboxes_to_dataset(
        dataset=dataset,
        pose_root=pose_root,
        conf_threshold=args.conf_threshold,
        margin=args.margin,
        min_valid_points=args.min_valid_points,
        clip_to_original_bbox=not args.no_clip_to_original_bbox,
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(enriched_dataset, f, indent=2)

    print(f"Saved enriched JSON: {output_json_path}")
    print("Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()