"""
Feature extraction and fusion utilities for I3D + TGCN concatenation.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


def load_pose_sequence(video_id, pose_root, num_frames=50):
    """
    Load and process OpenPose keypoints for one video.

    Returns tensor of shape (55, num_frames*2) — x and y coordinates
    for 55 keypoints across num_frames uniformly sampled frames.
    Returns None if the video folder has no keypoint files.
    """
    body_excl = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    vid_dir = Path(pose_root) / video_id
    frames = sorted(vid_dir.glob("image_*_keypoints.json"))
    if not frames:
        return None

    total = len(frames)
    idxs = [int(i * total / num_frames) for i in range(num_frames)]
    xy_frames = []

    for idx in idxs:
        try:
            c = json.loads(frames[idx].read_text())["people"][0]
        except (IndexError, KeyError):
            xy_frames.append(None)
            continue
        kpts = (c["pose_keypoints_2d"]
                + c["hand_left_keypoints_2d"]
                + c["hand_right_keypoints_2d"])
        x = torch.FloatTensor([v for i, v in enumerate(kpts) if i % 3 == 0 and i // 3 not in body_excl])
        y = torch.FloatTensor([v for i, v in enumerate(kpts) if i % 3 == 1 and i // 3 not in body_excl])
        x = 2 * (x / 256.0 - 0.5)
        y = 2 * (y / 256.0 - 0.5)
        xy_frames.append(torch.stack([x, y], dim=1))  # (55, 2)

    # Fill None entries with nearest valid frame
    for i in range(len(xy_frames)):
        if xy_frames[i] is None:
            for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(xy_frames))):
                if xy_frames[j] is not None:
                    xy_frames[i] = xy_frames[j]
                    break

    if any(f is None for f in xy_frames):
        return None

    return torch.cat(xy_frames, dim=1)  # (55, num_frames*2)


def extract_tgcn_features(tgcn, glosses_json, pose_root, num_frames=50):
    """
    Extract TGCN pose features for every video in glosses_json.

    Args:
        tgcn: GCN_muti_att model with extract_features() method, in eval mode
        glosses_json: path to glosses_valid.json
        pose_root: path to pose_per_individual_videos/
        num_frames: number of frames to sample per video

    Returns:
        features: {video_id: tensor(hidden_dim)}
        labels:   {video_id: class_index}
        splits:   {video_id: split_name}
    """
    data = json.loads(Path(glosses_json).read_text())
    features, labels, splits = {}, {}, {}

    with torch.no_grad():
        for class_id, entry in enumerate(data):
            for inst in entry["instances"]:
                vid = inst["video_id"]
                seq = load_pose_sequence(vid, pose_root, num_frames)
                if seq is None:
                    print(f"  Sin poses: {vid}")
                    continue
                feat = tgcn.extract_features(seq.unsqueeze(0)).squeeze(0)
                features[vid] = feat
                labels[vid]   = class_id
                splits[vid]   = inst["split"]

    print(f"Features TGCN extraidos: {len(features)} videos (dim={next(iter(features.values())).shape[0]})")
    return features, labels, splits


def extract_i3d_features(i3d, nslt_json, videos_dir, device):
    """
    Extract I3D video features for every video in nslt_json.

    Args:
        i3d: InceptionI3d model in eval mode, already on device
        nslt_json: path to nslt_valid.json
        videos_dir: path to folder with .mp4 files
        device: torch.device

    Returns:
        features: {video_id: tensor(1024)}
        labels:   {video_id: class_index}
        splits:   {video_id: split_name}
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

    for split_name, nslt_split in [("train", "train"), ("test", "test")]:
        ds = Dataset(str(nslt_json), split_name, root, "rgb", val_tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        with torch.no_grad():
            for inputs, lbls, vids in dl:
                inputs = inputs.to(device)
                feats  = i3d.extract_features(inputs).mean(dim=[2, 3, 4]).cpu()  # (B, 1024)
                gts    = torch.argmax(torch.max(lbls, dim=2)[0], dim=1)
                for vid, feat, gt in zip(vids, feats, gts):
                    features[vid] = feat
                    labels[vid]   = gt.item()
                    splits[vid]   = nslt_split

    print(f"Features I3D extraidos: {len(features)} videos (dim={next(iter(features.values())).shape[0]})")
    return features, labels, splits


def build_split_tensors(common_vids, tgcn_features, i3d_features, labels, splits):
    """
    Build train/test feature tensors from the common video set.

    train = subsets 'train' and 'val'; test = subset 'test'.

    Returns:
        (Xtr_tgcn, Xtr_i3d, Xtr_fuse, ytr,
         Xte_tgcn, Xte_i3d, Xte_fuse, yte)
    """
    train_vids = [v for v in common_vids if splits[v] in ("train", "val")]
    test_vids  = [v for v in common_vids if splits[v] == "test"]

    def _pack(vids):
        Xt = torch.stack([tgcn_features[v] for v in vids])
        Xi = torch.stack([i3d_features[v]  for v in vids])
        y  = torch.tensor([labels[v]        for v in vids])
        return Xt, Xi, torch.cat([Xt, Xi], dim=1), y

    Xtr_t, Xtr_i, Xtr_f, ytr = _pack(train_vids)
    Xte_t, Xte_i, Xte_f, yte = _pack(test_vids)

    print(f"Train: {len(train_vids)} | Test: {len(test_vids)}")
    return Xtr_t, Xtr_i, Xtr_f, ytr, Xte_t, Xte_i, Xte_f, yte


class ProjectionFusion(nn.Module):
    """
    Opción 2 — Proyección a espacio compartido + suma.

    Ambos vectores se proyectan a `shared_dim` con capas lineales
    independientes y se suman elemento a elemento antes del clasificador.
    Esto fuerza a que las dos modalidades contribuyan simétricamente,
    independientemente de su dimensión original.
    """
    def __init__(self, tgcn_dim, i3d_dim, shared_dim, num_classes, dropout=0.3):
        super().__init__()
        self.proj_tgcn  = nn.Linear(tgcn_dim, shared_dim)
        self.proj_i3d   = nn.Linear(i3d_dim,  shared_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

    def forward(self, x_tgcn, x_i3d):
        return self.classifier(self.proj_tgcn(x_tgcn) + self.proj_i3d(x_i3d))


class BilinearFusion(nn.Module):
    """
    Opción 4 — Bilinear pooling compacto (producto de Hadamard).

    Ambos vectores se proyectan a `latent_dim` y se multiplican elemento
    a elemento. El producto captura interacciones multiplicativas entre
    modalidades que la suma y la concatenación no pueden representar.
    """
    def __init__(self, tgcn_dim, i3d_dim, latent_dim, num_classes, dropout=0.3):
        super().__init__()
        self.proj_tgcn  = nn.Linear(tgcn_dim, latent_dim)
        self.proj_i3d   = nn.Linear(i3d_dim,  latent_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x_tgcn, x_i3d):
        return self.classifier(self.proj_tgcn(x_tgcn) * self.proj_i3d(x_i3d))


def _train_model(model, X_tr, y_tr, X_te, y_te, name, epochs, lr,
                 dual_input=False):
    """Bucle de entrenamiento genérico. Soporta entrada simple o dual."""
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(*X_tr) if dual_input else model(X_tr)
        F.cross_entropy(out, y_tr).backward()
        opt.step()

        if ep % 30 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out_te   = model(*X_te) if dual_input else model(X_te)
                out_tr   = model(*X_tr) if dual_input else model(X_tr)
                acc      = (out_te.argmax(1) == y_te).float().mean().item()
                loss_val = F.cross_entropy(out_tr, y_tr).item()
            best_acc = max(best_acc, acc)
            print(f"  [{name}] ep={ep:3d}  loss={loss_val:.4f}  test_acc={acc:.3f}")

    return best_acc


def train_fusion_head(X_tr, y_tr, X_te, y_te, in_dim, num_classes,
                      name="concat", epochs=150, lr=1e-3, hidden=512, dropout=0.3):
    """
    Opción 1 — MLP sobre concatenación de features.
    Entrena y devuelve la mejor accuracy en test.
    """
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )
    return _train_model(model, X_tr, y_tr, X_te, y_te, name, epochs, lr)


def train_projection_fusion(X_tr_tgcn, X_tr_i3d, y_tr,
                            X_te_tgcn, X_te_i3d, y_te,
                            tgcn_dim, i3d_dim, num_classes,
                            name="proj", shared_dim=512, epochs=150, lr=1e-3, dropout=0.3):
    """
    Opción 2 — Proyección a espacio compartido + suma.
    Entrena y devuelve la mejor accuracy en test.
    """
    model = ProjectionFusion(tgcn_dim, i3d_dim, shared_dim, num_classes, dropout)
    return _train_model(model, (X_tr_tgcn, X_tr_i3d), y_tr,
                        (X_te_tgcn, X_te_i3d), y_te, name, epochs, lr,
                        dual_input=True)


def train_bilinear_fusion(X_tr_tgcn, X_tr_i3d, y_tr,
                          X_te_tgcn, X_te_i3d, y_te,
                          tgcn_dim, i3d_dim, num_classes,
                          name="bilinear", latent_dim=512, epochs=150, lr=1e-3, dropout=0.3):
    """
    Opción 4 — Bilinear pooling compacto (producto de Hadamard).
    Entrena y devuelve la mejor accuracy en test.
    """
    model = BilinearFusion(tgcn_dim, i3d_dim, latent_dim, num_classes, dropout)
    return _train_model(model, (X_tr_tgcn, X_tr_i3d), y_tr,
                        (X_te_tgcn, X_te_i3d), y_te, name, epochs, lr,
                        dual_input=True)


# ── Fusión trimodal: TGCN + I3D + Boca ────────────────────────────────────────

def build_trimodal_split_tensors(common_vids, tgcn_features, i3d_features,
                                 mouth_features, labels, splits):
    """
    Como build_split_tensors pero incluye una 3ra modalidad (boca).
    Solo usa videos presentes en las 3 fuentes.

    Returns:
        (Xtr_t, Xtr_i, Xtr_m, Xtr_f, ytr,
         Xte_t, Xte_i, Xte_m, Xte_f, yte)
        donde Xtr_f = concat([Xtr_t, Xtr_i, Xtr_m]) ya listo para train_fusion_head.
    """
    train_vids = [v for v in common_vids if splits[v] in ("train", "val")]
    test_vids  = [v for v in common_vids if splits[v] == "test"]

    def _pack(vids):
        Xt = torch.stack([tgcn_features[v]  for v in vids])
        Xi = torch.stack([i3d_features[v]   for v in vids])
        Xm = torch.stack([mouth_features[v] for v in vids])
        y  = torch.tensor([labels[v]        for v in vids])
        return Xt, Xi, Xm, torch.cat([Xt, Xi, Xm], dim=1), y

    Xtr_t, Xtr_i, Xtr_m, Xtr_f, ytr = _pack(train_vids)
    Xte_t, Xte_i, Xte_m, Xte_f, yte = _pack(test_vids)

    print(f"Trimodal — Train: {len(train_vids)} | Test: {len(test_vids)}")
    return Xtr_t, Xtr_i, Xtr_m, Xtr_f, ytr, Xte_t, Xte_i, Xte_m, Xte_f, yte


class TrimodalProjectionFusion(nn.Module):
    """
    Proyección a espacio compartido + suma, extendida a 3 modalidades.

    Cada modalidad se proyecta a shared_dim con su propia capa lineal;
    las 3 proyecciones se suman antes del clasificador.
    """
    def __init__(self, tgcn_dim, i3d_dim, mouth_dim, shared_dim, num_classes, dropout=0.3):
        super().__init__()
        self.proj_tgcn  = nn.Linear(tgcn_dim,  shared_dim)
        self.proj_i3d   = nn.Linear(i3d_dim,   shared_dim)
        self.proj_mouth = nn.Linear(mouth_dim,  shared_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

    def forward(self, x_tgcn, x_i3d, x_mouth):
        fused = self.proj_tgcn(x_tgcn) + self.proj_i3d(x_i3d) + self.proj_mouth(x_mouth)
        return self.classifier(fused)


class TrimodalBilinearFusion(nn.Module):
    """
    Bilinear compacto (Hadamard), extendido a 3 modalidades.

    Las 3 modalidades se proyectan a latent_dim y se multiplican
    elemento a elemento: z = proj_t ⊙ proj_i ⊙ proj_m.
    """
    def __init__(self, tgcn_dim, i3d_dim, mouth_dim, latent_dim, num_classes, dropout=0.3):
        super().__init__()
        self.proj_tgcn  = nn.Linear(tgcn_dim,  latent_dim)
        self.proj_i3d   = nn.Linear(i3d_dim,   latent_dim)
        self.proj_mouth = nn.Linear(mouth_dim,  latent_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x_tgcn, x_i3d, x_mouth):
        fused = self.proj_tgcn(x_tgcn) * self.proj_i3d(x_i3d) * self.proj_mouth(x_mouth)
        return self.classifier(fused)


def _train_model_triple(model, X_tr, y_tr, X_te, y_te, name, epochs, lr):
    """Bucle de entrenamiento para modelos con 3 entradas."""
    opt      = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(*X_tr)
        F.cross_entropy(out, y_tr).backward()
        opt.step()

        if ep % 30 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out_te   = model(*X_te)
                out_tr   = model(*X_tr)
                acc      = (out_te.argmax(1) == y_te).float().mean().item()
                loss_val = F.cross_entropy(out_tr, y_tr).item()
            best_acc = max(best_acc, acc)
            print(f"  [{name}] ep={ep:3d}  loss={loss_val:.4f}  test_acc={acc:.3f}")

    return best_acc


def train_trimodal_projection_fusion(
        Xtr_t, Xtr_i, Xtr_m, ytr,
        Xte_t, Xte_i, Xte_m, yte,
        tgcn_dim, i3d_dim, mouth_dim, num_classes,
        name="Proj+Suma+Boca", shared_dim=512, epochs=150, lr=1e-3, dropout=0.3):
    model = TrimodalProjectionFusion(tgcn_dim, i3d_dim, mouth_dim,
                                     shared_dim, num_classes, dropout)
    return _train_model_triple(model,
                               (Xtr_t, Xtr_i, Xtr_m), ytr,
                               (Xte_t, Xte_i, Xte_m), yte,
                               name, epochs, lr)


def train_trimodal_bilinear_fusion(
        Xtr_t, Xtr_i, Xtr_m, ytr,
        Xte_t, Xte_i, Xte_m, yte,
        tgcn_dim, i3d_dim, mouth_dim, num_classes,
        name="Bilinear+Boca", latent_dim=512, epochs=150, lr=1e-3, dropout=0.3):
    model = TrimodalBilinearFusion(tgcn_dim, i3d_dim, mouth_dim,
                                   latent_dim, num_classes, dropout)
    return _train_model_triple(model,
                               (Xtr_t, Xtr_i, Xtr_m), ytr,
                               (Xte_t, Xte_i, Xte_m), yte,
                               name, epochs, lr)
