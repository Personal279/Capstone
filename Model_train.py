"""
train.py

Experiment C only (A and B removed):
  C: ResNet50 + FAI as an AUXILIARY TRAINING SIGNAL (not a classifier input --
     see ResNetFAIAux docstring below for why direct fusion was rejected)
     + legitimate training improvements (class weights, label smoothing,
     LR scheduler, discriminative LR, dropout/weight decay tuning).

IMPORTANT ARCHITECTURE NOTE:
An earlier version of this script concatenated the raw 5 FAI values directly
into the classifier. Because those same 5 values are what your labeling
script used to compute the pseudo-label (via nearest-anchor distance), that
let the model reconstruct the labeling rule instead of learning from the
image -- it produced an unrealistic ~0.97 val kappa by epoch 2, which is a
leakage artifact, not real performance. This version fixes that: FAI values
are used ONLY as an auxiliary regression target during training (multi-task
learning), never concatenated into the classification head. Predictions at
both train and test time come from the image alone.

Checkpoint selection uses VALIDATION accuracy only. The validation split is
carved out of TRAIN patients only (never touches test patients). TEST is
evaluated exactly once, at the very end, using the checkpoint chosen on
validation.

Expected CSV format: your labeling script's raw output CSV works directly --
only image_path, patient_id, label, no_face_detected are used. FAI features
are computed on the fly (once, cached in memory) using fai_extract.py, which
is an exact port of your labeling script's extract_features() -- no separate
precomputed CSV needed.

`label` is 1-5 (string or int). `patient_id` must be present and consistent
per patient so the val split can be carved out without leaking patients
across train/val.

Usage:
    python train.py --train_csv train_clean.csv --test_csv test_clean.csv
"""

import os

# Must be set before importing cv2/mediapipe -- on this many-core ARM box,
# per-library thread pools (OpenCV, TFLite/XNNPACK) have been observed to
# intermittently deadlock (futex hang, zero CPU progress) during a long
# single-process run over thousands of images. This mitigates it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import argparse
import random
import numpy as np
import cv2
cv2.setNumThreads(1)
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score
)

NUM_CLASSES = 5
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────── FAI FEATURE EXTRACTION ───────────────────────────
# Exact port of extract_features() from the labeling script -- same landmark
# indices, same pixel-space coordinates, same formulas. Do not alter this --
# it must stay bit-identical to whatever generated the pseudo-labels.

_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5,
)

_LEFT_EYE_OUTER, _RIGHT_EYE_OUTER = 33, 263
_LEFT_EYE_TOP, _LEFT_EYE_BOTTOM = 159, 145
_RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM = 386, 374
_LEFT_BROW, _RIGHT_BROW = 105, 334
_NOSE_BRIDGE, _CHIN = 6, 152
_MOUTH_LEFT, _MOUTH_RIGHT = 61, 291
_MOUTH_TOP, _MOUTH_BOTTOM = 13, 14


def _cross_2d(a, b):
    """Manual 2D cross product (scalar) -- np.cross() dropped 2D vector
    support in NumPy 2.0+, this is the mathematically identical replacement."""
    return a[0] * b[1] - a[1] * b[0]


def extract_fai_features(image_path):
    """Returns np.ndarray shape (5,) or None if no usable face detected."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    def pt(index):
        return np.array([landmarks[index].x * w, landmarks[index].y * h], dtype=np.float32)

    left_eye_outer = pt(_LEFT_EYE_OUTER)
    right_eye_outer = pt(_RIGHT_EYE_OUTER)
    inter_ocular = np.linalg.norm(left_eye_outer - right_eye_outer)
    if inter_ocular < 1e-6:
        return None

    # Eye asymmetry
    left_eye_open = np.linalg.norm(pt(_LEFT_EYE_TOP) - pt(_LEFT_EYE_BOTTOM)) / inter_ocular
    right_eye_open = np.linalg.norm(pt(_RIGHT_EYE_TOP) - pt(_RIGHT_EYE_BOTTOM)) / inter_ocular
    eye_asymmetry = abs(left_eye_open - right_eye_open)

    # Eyebrow asymmetry (signed heights, differenced -- matches labeling script)
    left_brow_height = (pt(_LEFT_BROW)[1] - pt(_LEFT_EYE_TOP)[1]) / inter_ocular
    right_brow_height = (pt(_RIGHT_BROW)[1] - pt(_RIGHT_EYE_TOP)[1]) / inter_ocular
    brow_asymmetry = abs(left_brow_height - right_brow_height)

    # Mouth asymmetry relative to nose-chin midline
    nose = pt(_NOSE_BRIDGE)
    chin = pt(_CHIN)
    midline = chin - nose
    midline_norm = np.linalg.norm(midline)
    if midline_norm < 1e-6:
        return None
    midline_unit = midline / midline_norm

    mouth_left = pt(_MOUTH_LEFT)
    mouth_right = pt(_MOUTH_RIGHT)
    left_dist = abs(_cross_2d(midline_unit, mouth_left - nose))
    right_dist = abs(_cross_2d(midline_unit, mouth_right - nose))
    mouth_asymmetry = abs(left_dist - right_dist) / inter_ocular

    # Mouth vertical asymmetry
    mouth_vertical_asymmetry = abs(mouth_left[1] - mouth_right[1]) / inter_ocular

    # Mouth openness
    mouth_open = np.linalg.norm(pt(_MOUTH_TOP) - pt(_MOUTH_BOTTOM)) / inter_ocular

    features = np.array([
        eye_asymmetry, brow_asymmetry, mouth_asymmetry,
        mouth_vertical_asymmetry, mouth_open,
    ], dtype=np.float32)

    if not np.all(np.isfinite(features)):
        return None
    return features


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────── DATA ───────────────────────────

def load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader
                if r.get("label") in {"1", "2", "3", "4", "5"}
                and r.get("no_face_detected", "False") != "True"]
    return rows


def patient_wise_val_split(train_rows, val_fraction=0.15, seed=42):
    """Carve a validation set out of TRAIN patients only. Test patients are
    never touched here -- this function never sees test_rows."""
    rng = random.Random(seed)
    patient_ids = sorted(set(r["patient_id"] for r in train_rows))
    rng.shuffle(patient_ids)
    n_val_patients = max(1, int(len(patient_ids) * val_fraction))
    val_patients = set(patient_ids[:n_val_patients])

    val_rows = [r for r in train_rows if r["patient_id"] in val_patients]
    fit_rows = [r for r in train_rows if r["patient_id"] not in val_patients]
    return fit_rows, val_rows


class SeverityDataset(Dataset):
    """Returns (image_tensor, fai_tensor, label). FAI features are computed
    ONCE per row at construction time (via fai_extract.extract_features --
    an exact port of the labeling script's formula) and cached in memory,
    so MediaPipe doesn't re-run every epoch. Rows where re-extraction fails
    (rare -- face detected during labeling but not now) are dropped, with
    a count printed so silent data loss doesn't go unnoticed."""

    def __init__(self, rows, transform, image_root="", desc="dataset"):
        self.transform = transform
        self.image_root = image_root
        self.samples = []  # list of (image_path, fai_array, label)

        n_failed = 0
        for row in rows:
            img_path = os.path.join(image_root, row["image_path"])
            fai = extract_fai_features(img_path)
            if fai is None:
                n_failed += 1
                continue
            label = int(row["label"]) - 1
            self.samples.append((img_path, fai, label))

        print(f"[{desc}] {len(self.samples)} usable images "
              f"({n_failed} skipped -- FAI re-extraction failed)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fai, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.from_numpy(fai), label


def build_transforms():
    # No horizontal flip: flipping mirrors which side is paralyzed and would
    # corrupt the asymmetry signal both the FAI branch and ResNet rely on.
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# ─────────────────────────── MODEL ───────────────────────────

class ResNetFAIAux(nn.Module):
    """FAI as an AUXILIARY TRAINING SIGNAL, not a classifier input.

    IMPORTANT / WHY THIS ARCHITECTURE CHANGED:
    The pseudo-labels were generated via nearest-anchor distance computed
    directly from these same 5 FAI features. Concatenating the raw FAI
    values into the classifier (an earlier design) let the model
    reconstruct the labeling rule itself instead of learning to classify
    severity from the image -- that produced an unrealistic ~0.97 val kappa
    by epoch 2, which is a leakage artifact, not real performance.

    This version removes that leak: the classification head sees ONLY the
    visual embedding, at both train and test time -- FAI values are never
    concatenated into it. FAI is used solely as an auxiliary regression
    TARGET during training (predict the 5 FAI values from the image), which
    forces the visual backbone to encode asymmetry-relevant features without
    ever handing the classifier the exact numbers that generated the label.
    The auxiliary head is discarded at inference; only the classification
    head's output is used for predictions/metrics.
    """
    def __init__(self, num_classes=NUM_CLASSES, fai_mean=None, fai_std=None, dropout=0.4):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.visual_features = nn.Sequential(*list(backbone.children())[:-1])
        visual_dim = backbone.fc.in_features  # 2048

        self.trunk = nn.Sequential(
            nn.Linear(visual_dim, 256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier_head = nn.Linear(256, num_classes)
        self.fai_aux_head = nn.Linear(256, 5)  # predicts standardized FAI values

        self.register_buffer("fai_mean", torch.tensor(fai_mean if fai_mean is not None else [0.0]*5))
        self.register_buffer("fai_std", torch.tensor(fai_std if fai_std is not None else [1.0]*5))

    def forward(self, image, fai=None):
        v = self.visual_features(image).flatten(1)
        v = self.trunk(v)
        logits = self.classifier_head(v)
        fai_pred = self.fai_aux_head(v)  # standardized-scale prediction
        return logits, fai_pred


def build_model(fai_mean=None, fai_std=None):
    return ResNetFAIAux(fai_mean=fai_mean, fai_std=fai_std, dropout=0.4).to(DEVICE)


# ─────────────────────────── TRAIN / EVAL ───────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, aux_weight=0.3):
    model.train()
    total_loss = 0.0
    mse = nn.MSELoss()
    for images, fai, labels in loader:
        images, fai, labels = images.to(DEVICE), fai.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        logits, fai_pred = model(images)
        fai_target = (fai - model.fai_mean) / (model.fai_std + 1e-6)
        cls_loss = criterion(logits, labels)
        aux_loss = mse(fai_pred, fai_target)
        loss = cls_loss + aux_weight * aux_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, fai, labels in loader:
        images = images.to(DEVICE)
        outputs, _ = model(images)  # discard auxiliary FAI prediction -- never used for classification
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred):
    exact_acc = accuracy_score(y_true, y_pred)
    within_one = float(np.mean(np.abs(y_true - y_pred) <= 1))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return {
        "exact_acc": exact_acc, "within_one_acc": within_one, "macro_f1": macro_f1,
        "kappa": kappa, "precision": precision, "recall": recall, "f1": f1,
        "support": support, "confusion_matrix": cm,
    }


def print_metrics(tag, m):
    print(f"\n{'='*55}\n{tag}\n{'='*55}")
    print(f"Exact-match accuracy:       {m['exact_acc']*100:.2f}%")
    print(f"+/-1-grade accuracy:        {m['within_one_acc']*100:.2f}%")
    print(f"Macro F1:                   {m['macro_f1']:.4f}")
    print(f"Quadratic weighted kappa:   {m['kappa']:.4f}")
    print(f"\n{'Grade':<8}{'Precision':<12}{'Recall':<10}{'F1':<10}{'Support':<10}")
    for i in range(NUM_CLASSES):
        print(f"{i+1:<8}{m['precision'][i]:<12.3f}{m['recall'][i]:<10.3f}{m['f1'][i]:<10.3f}{m['support'][i]:<10}")
    print("\nConfusion matrix (rows=true, cols=predicted, grades 1-5):")
    print(m["confusion_matrix"])


def build_optimizer(model, lr_backbone, lr_head):
    backbone_params = model.visual_features.parameters()
    head_params = list(model.trunk.parameters()) + \
                  list(model.classifier_head.parameters()) + \
                  list(model.fai_aux_head.parameters())
    return torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ])


def run_experiment(train_csv, test_csv, image_root, epochs,
                    batch_size, patience, use_class_weights=True, label_smoothing=0.05):
    print(f"\n{'#'*60}\n# EXPERIMENT C\n{'#'*60}")
    set_seed(42)

    all_train_rows = load_rows(train_csv)
    test_rows = load_rows(test_csv)  # touched only at the very end
    fit_rows, val_rows = patient_wise_val_split(all_train_rows, val_fraction=0.15, seed=42)

    train_patients = set(r["patient_id"] for r in fit_rows)
    val_patients = set(r["patient_id"] for r in val_rows)
    test_patients = set(r["patient_id"] for r in test_rows)
    assert train_patients.isdisjoint(val_patients), "Patient leak: train/val"
    assert train_patients.isdisjoint(test_patients), "Patient leak: train/test"
    assert val_patients.isdisjoint(test_patients), "Patient leak: val/test"

    print(f"Fit: {len(fit_rows)} images / {len(train_patients)} patients")
    print(f"Val: {len(val_rows)} images / {len(val_patients)} patients")
    print(f"Test: {len(test_rows)} images / {len(test_patients)} patients")

    train_tf, eval_tf = build_transforms()
    # FAI features computed once here (cached inside each Dataset). This is
    # the slow step (MediaPipe per image) -- runs once per run.
    fit_ds = SeverityDataset(fit_rows, train_tf, image_root, desc="fit")
    val_ds = SeverityDataset(val_rows, eval_tf, image_root, desc="val")
    test_ds = SeverityDataset(test_rows, eval_tf, image_root, desc="test")

    # FAI normalization stats computed from FIT split only (never val/test)
    fit_fai_matrix = np.array([s[1] for s in fit_ds.samples])
    fai_mean = fit_fai_matrix.mean(axis=0).tolist()
    fai_std = fit_fai_matrix.std(axis=0).tolist()

    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(fai_mean=fai_mean, fai_std=fai_std)

    if use_class_weights:
        labels_fit = np.array([s[2] for s in fit_ds.samples])
        counts = np.bincount(labels_fit, minlength=NUM_CLASSES)
        weights = (counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)))
        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    lr_backbone = 1e-5
    lr_head = 1e-3
    optimizer = build_optimizer(model, lr_backbone, lr_head)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    ckpt_path = "best_model_experiment_C.pt"
    best_val_kappa = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, fit_loader, optimizer, criterion)
        y_val_true, y_val_pred = evaluate(model, val_loader)
        val_metrics = compute_metrics(y_val_true, y_val_pred)
        val_acc, val_kappa = val_metrics["exact_acc"], val_metrics["kappa"]

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val_acc*100:.2f}% | val_kappa={val_kappa:.4f}")

        scheduler.step(val_kappa)

        # Selection metric: quadratic kappa (matches the ordinal nature of the
        # task better than raw accuracy). Selected and saved using VAL ONLY.
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best (val_kappa={val_kappa:.4f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"  -> early stopping (no val improvement for {patience} epochs)")
                break

    # TEST touched exactly once, using the checkpoint selected on VAL
    model.load_state_dict(torch.load(ckpt_path))
    y_test_true, y_test_pred = evaluate(model, test_loader)
    test_metrics = compute_metrics(y_test_true, y_test_pred)
    print_metrics("Experiment C — FINAL TEST RESULTS", test_metrics)
    return test_metrics


# ─────────────────────────── HARDCODED PATHS ───────────────────────────
# Edit these to point at your actual files -- no CLI args needed anymore.
TRAIN_CSV = "train_clean.csv"
TEST_CSV = "test_clean.csv"
IMAGE_ROOT = ""   # base folder prepended to each row's image_path, if any
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--test_csv", default=TEST_CSV)
    parser.add_argument("--image_root", default=IMAGE_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    run_experiment(
        args.train_csv, args.test_csv, args.image_root,
        args.epochs, args.batch_size, args.patience,
        use_class_weights=True, label_smoothing=0.05,
    )


if __name__ == "__main__":
    main()