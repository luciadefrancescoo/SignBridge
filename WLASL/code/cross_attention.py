"""
Cross-attention fusion of I3D (video) and TGCN (pose) sequence features.

Unlike the strategies in concat.py (which pool each modality to a single
vector before fusing), this module keeps sequences intact so that the
attention mechanism can learn *which frames* attend to *which keypoints*:

  I3D  → (B, T,  1024)   spatial mean-pooled temporal features
  TGCN → (B, 55, 256)    per-keypoint features (before graph mean-pool)

Two cross-attention passes run in parallel:
  video-to-pose  Q=video,  K=V=pose   each frame queries the skeleton
  pose-to-video  Q=pose,   K=V=video  each keypoint queries the video

Both outputs are mean-pooled and concatenated before the MLP classifier.

Expected gains over simple concatenation on large datasets:
  - Captures frame-level pose correlations (e.g. handshape at frame t)
  - Symmetric: neither modality dominates by vector size
  - Scales well with data; attention weights are interpretable
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_tgcn_seq_features(tgcn, glosses_json, pose_root, num_frames=50):
    """
    Extract per-keypoint TGCN features: Tensor(55, hidden_dim) per video.

    This is the pre-pool version of concat.extract_tgcn_features — it stops
    before the mean(dim=1) call in GCN_muti_att._backbone so each of the 55
    body keypoints keeps its own feature vector.

    Args:
        tgcn:        GCN_muti_att model, eval mode
        glosses_json: path to glosses_valid.json (top-N subset)
        pose_root:   path to pose_per_individual_videos/
        num_frames:  frames to sample per video (must match TGCN input_feature // 2)

    Returns:
        features: {video_id: Tensor(55, hidden_dim)}
        labels:   {video_id: int}
        splits:   {video_id: str}
    """
    from concat import load_pose_sequence

    data = json.loads(Path(glosses_json).read_text())
    features, labels, splits = {}, {}, {}

    tgcn.eval()
    with torch.no_grad():
        for class_id, entry in enumerate(data):
            for inst in entry["instances"]:
                vid = inst["video_id"]
                seq = load_pose_sequence(vid, pose_root, num_frames)
                if seq is None:
                    continue

                x = seq.unsqueeze(0)                             # (1, 55, num_frames*2)
                y = tgcn.gc1(x)
                b, n, f = y.shape
                y = tgcn.bn1(y.view(b, -1)).view(b, n, f)
                y = tgcn.act_f(y)
                y = tgcn.do(y)
                for gcb in tgcn.gcbs:
                    y = gcb(y)
                # y: (1, 55, hidden_dim) — do NOT mean-pool here
                features[vid] = y.squeeze(0).cpu()
                labels[vid]   = class_id
                splits[vid]   = inst["split"]

    sample = next(iter(features.values()))
    print(f"TGCN seq features: {len(features)} videos — shape {tuple(sample.shape)}")
    return features, labels, splits


def extract_i3d_seq_features(i3d, nslt_json, videos_dir, device, t_max=16):
    """
    Extract spatial-pooled temporal I3D features: Tensor(t_max, 1024) per video.

    i3d.extract_features returns (B, 1024, T, H, W); we average over H and W
    to get (B, 1024, T), then transpose to (B, T, 1024). Videos are
    truncated or zero-padded to t_max so they can be stacked into a batch.

    Args:
        i3d:       InceptionI3d in eval mode, on device
        nslt_json: path to nslt_valid.json
        videos_dir: directory with .mp4 files
        device:    torch.device
        t_max:     fixed temporal length after padding/truncation

    Returns:
        features: {video_id: Tensor(t_max, 1024)}
        labels:   {video_id: int}
        splits:   {video_id: str}
    """
    i3d_dir = Path(__file__).parent / "I3D"
    for p in [str(i3d_dir), str(i3d_dir / "datasets")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import videotransforms
    from datasets.nslt_dataset import NSLT as Dataset

    val_tf = transforms.Compose([videotransforms.CenterCrop(224)])
    root   = {"word": str(videos_dir)}
    features, labels, splits = {}, {}, {}

    i3d.eval()
    for split_name in ["train", "test"]:
        ds = Dataset(str(nslt_json), split_name, root, "rgb", val_tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        with torch.no_grad():
            for inputs, lbls, vids in dl:
                raw  = i3d.extract_features(inputs.to(device))  # (B, 1024, T, H, W)
                raw  = raw.mean(dim=[-2, -1]).permute(0, 2, 1).cpu()  # (B, T, 1024)
                gts  = torch.argmax(torch.max(lbls, dim=2)[0], dim=1)

                for vid, feat, gt in zip(vids, raw, gts):
                    T = feat.shape[0]
                    if T >= t_max:
                        feat = feat[:t_max]
                    else:
                        feat = torch.cat([feat, torch.zeros(t_max - T, feat.shape[1])], dim=0)
                    features[vid] = feat                         # (t_max, 1024)
                    labels[vid]   = gt.item()
                    splits[vid]   = split_name

    sample = next(iter(features.values()))
    print(f"I3D seq features: {len(features)} videos — shape {tuple(sample.shape)}")
    return features, labels, splits


def build_seq_split_tensors(common_vids, tgcn_seq, i3d_seq, labels, splits):
    """
    Stack per-video sequence tensors into batched train/test sets.
    train = 'train' + 'val', test = 'test'  (same convention as concat.py).

    Returns:
        Xtr_video (N_tr, T,  1024)
        Xtr_pose  (N_tr, 55, hidden_dim)
        ytr       (N_tr,)
        Xte_video (N_te, T,  1024)
        Xte_pose  (N_te, 55, hidden_dim)
        yte       (N_te,)
    """
    train_vids = [v for v in common_vids if splits[v] in ("train", "val")]
    test_vids  = [v for v in common_vids if splits[v] == "test"]

    def _pack(vids):
        return (
            torch.stack([i3d_seq[v]  for v in vids]),
            torch.stack([tgcn_seq[v] for v in vids]),
            torch.tensor([labels[v]  for v in vids]),
        )

    Xtr_v, Xtr_p, ytr = _pack(train_vids)
    Xte_v, Xte_p, yte = _pack(test_vids)
    print(f"Train: {len(train_vids)} | Test: {len(test_vids)}")
    return Xtr_v, Xtr_p, ytr, Xte_v, Xte_p, yte


# ── Model ─────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion of video and pose sequences.

    Forward inputs:
        x_video: (B, T,  video_dim)   I3D temporal features
        x_pose:  (B, 55, pose_dim)    TGCN keypoint features

    Processing:
        1. Project both to d_model
        2. video-to-pose cross-attention  (each frame attends to keypoints)
        3. pose-to-video cross-attention  (each keypoint attends to frames)
        4. Residual + LayerNorm on each branch
        5. Mean-pool → concatenate → MLP

    Args:
        video_dim:   input dim of video features  (default 1024 for I3D)
        pose_dim:    input dim of pose features   (default 256 for TGCN)
        d_model:     shared attention dimension
        num_heads:   attention heads (d_model must be divisible by num_heads)
        num_classes: output classes
        dropout:     applied inside attention and MLP
    """

    def __init__(
        self,
        video_dim: int = 1024,
        pose_dim:  int = 256,
        d_model:   int = 256,
        num_heads: int = 4,
        num_classes: int = 20,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.proj_video = nn.Linear(video_dim, d_model)
        self.proj_pose  = nn.Linear(pose_dim,  d_model)

        mha = dict(embed_dim=d_model, num_heads=num_heads,
                   dropout=dropout, batch_first=True)
        self.attn_v2p = nn.MultiheadAttention(**mha)  # video queries pose
        self.attn_p2v = nn.MultiheadAttention(**mha)  # pose queries video

        self.norm_v = nn.LayerNorm(d_model)
        self.norm_p = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_video: torch.Tensor, x_pose: torch.Tensor) -> torch.Tensor:
        v = self.proj_video(x_video)   # (B, T,  d_model)
        p = self.proj_pose(x_pose)     # (B, 55, d_model)

        v_att, _ = self.attn_v2p(query=v, key=p, value=p)   # (B, T,  d_model)
        p_att, _ = self.attn_p2v(query=p, key=v, value=v)   # (B, 55, d_model)

        v_out = self.norm_v(v + v_att).mean(dim=1)   # (B, d_model)
        p_out = self.norm_p(p + p_att).mean(dim=1)   # (B, d_model)

        return self.classifier(torch.cat([v_out, p_out], dim=1))


# ── Training ──────────────────────────────────────────────────────────────────

def train_cross_attention(
    Xtr_video, Xtr_pose, ytr,
    Xte_video, Xte_pose, yte,
    video_dim:   int   = 1024,
    pose_dim:    int   = 256,
    num_classes: int   = 20,
    d_model:     int   = 256,
    num_heads:   int   = 4,
    epochs:      int   = 200,
    lr:          float = 1e-3,
    batch_size:  int   = 32,
    dropout:     float = 0.3,
    name:        str   = "CrossAttn",
) -> float:
    """
    Train CrossAttentionFusion with mini-batches and return best test accuracy.

    Mini-batch training is important here: the full dataset doesn't fit in GPU
    memory as a single forward pass, unlike the small top-20 subset.

    Returns:
        best test accuracy (float)
    """
    model = CrossAttentionFusion(video_dim, pose_dim, d_model,
                                 num_heads, num_classes, dropout)
    opt  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    N    = len(ytr)
    best = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        ep_loss = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            out  = model(Xtr_video[idx], Xtr_pose[idx])
            loss = F.cross_entropy(out, ytr[idx])
            loss.backward()
            opt.step()
            opt.zero_grad()
            ep_loss += loss.item()

        if ep % 50 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out_te = model(Xte_video, Xte_pose)
                acc    = (out_te.argmax(1) == yte).float().mean().item()
            best = max(best, acc)
            print(f"  [{name}] ep={ep:3d}  loss={ep_loss / max(N // batch_size, 1):.4f}"
                  f"  test_acc={acc:.3f}")

    return best
