"""
Mouth crop extraction and encoding using MediaPipe FaceLandmarker + MobileNetV2.

Extraction pipeline per video:
  1. Sample N frames uniformly from the video
  2. Run FaceLandmarker on each frame → 478 face landmarks
  3. Compute a padded bounding box from the lip landmark subset
  4. Crop and resize to crop_size × crop_size

Encoding pipeline (after extraction):
  5. Pass each crop through frozen MobileNetV2 (pretrained on ImageNet)
  6. Mean-pool the N frame features → one vector per video

Usage:
    extractor = MouthCropExtractor(model_path="face_landmarker.task")
    crops = extractor.extract_video("path/to/video.mp4")
    # crops: np.ndarray of shape (n_frames, crop_size, crop_size, 3) or None

    features = encode_mouth_crops(crops_dict)
    # features: {video_id: Tensor(1280)}
"""

import json
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# MediaPipe FaceMesh 478-point model — outer + inner lip contour indices
_LIP_IDX = frozenset({
    # outer
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # inner
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
})


class MouthCropExtractor:
    """
    Wraps MediaPipe FaceLandmarker for mouth region extraction.

    Args:
        model_path: path to face_landmarker.task file
        n_frames:   number of frames to sample uniformly per video
        crop_size:  side length (px) of the square output crop
        pad:        padding added around the tight lip bbox (px)
    """

    def __init__(
        self,
        model_path: str,
        n_frames:   int = 16,
        crop_size:  int = 64,
        pad:        int = 20,
    ):
        self.n_frames  = n_frames
        self.crop_size = crop_size
        self.pad       = pad

        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        opts = mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
        self._detector = mp_vision.FaceLandmarker.create_from_options(opts)

    def _lip_bbox(self, landmarks, frame_w: int, frame_h: int):
        """Return (x1, y1, x2, y2) padded bbox from lip landmarks, or None."""
        xs = [landmarks[i].x * frame_w for i in _LIP_IDX]
        ys = [landmarks[i].y * frame_h for i in _LIP_IDX]
        x1 = max(0,       int(min(xs)) - self.pad)
        y1 = max(0,       int(min(ys)) - self.pad)
        x2 = min(frame_w, int(max(xs)) + self.pad)
        y2 = min(frame_h, int(max(ys)) + self.pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def extract_video(self, video_path: str):
        """
        Extract mouth crops from one video.

        Returns:
            np.ndarray of shape (n_frames, crop_size, crop_size, 3)  uint8 RGB
            or None if fewer than half the frames have a detectable face.
        """
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return None

        # Uniform frame indices
        indices = [int(i * total / self.n_frames) for i in range(self.n_frames)]

        crops      = []
        last_bbox  = None
        miss_count = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                if last_bbox is not None:
                    crops.append(self._crop_frame(frame if ret else None, last_bbox))
                else:
                    miss_count += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result    = self._detector.detect(mp_img)

            if result.face_landmarks:
                h, w  = frame.shape[:2]
                bbox  = self._lip_bbox(result.face_landmarks[0], w, h)
                if bbox is not None:
                    last_bbox = bbox
                    crops.append(self._crop_frame(frame_rgb, bbox))
                    continue

            # No face detected: forward-fill from last known bbox
            if last_bbox is not None:
                crops.append(self._crop_frame(frame_rgb, last_bbox))
            else:
                miss_count += 1

        cap.release()

        if miss_count > self.n_frames // 2:
            return None

        # Back-fill any leading misses (rare: face not found in first frames)
        while len(crops) < self.n_frames:
            crops.insert(0, crops[0] if crops else np.zeros(
                (self.crop_size, self.crop_size, 3), dtype=np.uint8))

        return np.stack(crops[: self.n_frames])  # (n_frames, H, W, 3)

    def _crop_frame(self, frame_rgb, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame_rgb[y1:y2, x1:x2] if frame_rgb is not None else np.zeros(
            (y2 - y1, x2 - x1, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.crop_size, self.crop_size),
                          interpolation=cv2.INTER_LINEAR)


# ── Dataset-level extraction ──────────────────────────────────────────────────

def extract_all_mouth_crops(
    glosses_json: str,
    videos_dir:   str,
    model_path:   str,
    n_frames:     int = 16,
    crop_size:    int = 64,
    pad:          int = 20,
    save_path:    str = None,
):
    """
    Extract mouth crops for every video listed in glosses_json.

    Args:
        glosses_json: path to glosses_valid.json (top-N subset)
        videos_dir:   directory containing {video_id}.mp4 files
        model_path:   path to face_landmarker.task
        n_frames:     frames to sample per video
        crop_size:    output crop size in pixels
        pad:          padding around lip bbox
        save_path:    if given, saves result dict as a .pt file with torch.save

    Returns:
        crops:  {video_id: np.ndarray(n_frames, crop_size, crop_size, 3)}
        labels: {video_id: int}
        splits: {video_id: str}
    """
    extractor   = MouthCropExtractor(model_path, n_frames, crop_size, pad)
    data        = json.loads(Path(glosses_json).read_text())
    videos_dir  = Path(videos_dir)

    crops, labels, splits = {}, {}, {}
    ok = skip = 0

    for class_id, entry in enumerate(data):
        for inst in entry["instances"]:
            vid  = inst["video_id"]
            path = videos_dir / f"{vid}.mp4"
            if not path.exists():
                skip += 1
                continue

            result = extractor.extract_video(str(path))
            if result is None:
                print(f"  Sin cara detectada: {vid}")
                skip += 1
                continue

            crops[vid]  = result                   # (n_frames, H, W, 3)
            labels[vid] = class_id
            splits[vid] = inst["split"]
            ok += 1

    print(f"Mouth crops extraidos: {ok} videos | saltados: {skip}")

    if save_path is not None:
        import torch
        torch.save({"crops": crops, "labels": labels, "splits": splits},
                   save_path)
        print(f"Guardado en: {save_path}")

    return crops, labels, splits


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_mouth_crops(crops_dict, device=None, batch_size=32):
    """
    Encode mouth crops using frozen MobileNetV2 (pretrained on ImageNet).

    Each video's N frames are embedded individually, then mean-pooled to
    produce one 1280-dim feature vector per video.

    Args:
        crops_dict: {video_id: np.ndarray(n_frames, H, W, 3)}  uint8 RGB
        device:     torch.device (default: MPS → CPU fallback)
        batch_size: frames processed per forward pass

    Returns:
        {video_id: Tensor(1280)}
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    weights  = models.MobileNet_V2_Weights.IMAGENET1K_V1
    backbone = models.mobilenet_v2(weights=weights)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Drop classifier; pool spatial dims to 1×1 → (B, 1280)
    encoder = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ).to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    features = {}
    with torch.no_grad():
        for vid, crops in crops_dict.items():
            # crops: (N, H, W, 3) uint8 → float [0,1] → (N, 3, H, W)
            t = torch.from_numpy(crops).float().div(255.0)
            t = normalize(t.permute(0, 3, 1, 2))

            frame_feats = []
            for i in range(0, len(t), batch_size):
                frame_feats.append(encoder(t[i : i + batch_size].to(device)).cpu())

            features[vid] = torch.cat(frame_feats).mean(dim=0)  # (1280,)

    dim = next(iter(features.values())).shape[0]
    print(f"Mouth features encoded: {len(features)} videos — dim={dim}")
    return features
