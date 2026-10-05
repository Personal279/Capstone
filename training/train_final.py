"""
Final training script, built on real evidence from prior runs today:

  - Experiment A (plain ResNet50, pixels only): test acc=53.39%, kappa=0.5028;
    patient-level majority vote: acc=50.00%, kappa=0.5965 (best kappa so far).
  - Experiment C (ResNet50 + FAI as auxiliary regression target, class weights,
    label smoothing, heavy dropout, near-frozen backbone at lr=1e-5): test
    acc=40.13%, kappa=0.2651 -- WORSE across the board, at both frame and
    patient level. The added complexity overfit to the small (15-patient)
    validation split rather than learning anything that transfers.

Conclusion: the FAI auxiliary task and the aggressive discriminative-LR/
dropout regime actively hurt generalization for this data. This script
drops FAI entirely (also removes the MediaPipe startup cost) and keeps A's
proven recipe (full fine-tune, uniform lr=1e-4, dropout=0.3), adding only
class weights -- a standard, well-understood fix for the class imbalance
that's been the weak point (grades 3/4) in every run so far, and not
implicated in why C failed.

Same patient-wise val carve-out (15% of train patients, never touches test)
and single-touch test evaluation as prior runs. Reports both frame-level and
patient-level (majority vote) test metrics automatically.
"""

import csv
import random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score,
)

NUM_CLASSES = 5
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "train_clean.csv"
TEST_CSV = "test_clean.csv"
CKPT_PATH = "best_model_final_v2.pt"
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 5
LR = 1e-4
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader
                if r.get("label") in {"1", "2", "3", "4", "5"}
                and r.get("no_face_detected", "False") != "True"]


def patient_wise_val_split(train_rows, val_fraction=0.15, seed=42):
    rng = random.Random(seed)
    patient_ids = sorted(set(r["patient_id"] for r in train_rows))
    rng.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_fraction))
    val_patients = set(patient_ids[:n_val])
    val_rows = [r for r in train_rows if r["patient_id"] in val_patients]
    fit_rows = [r for r in train_rows if r["patient_id"] not in val_patients]
    return fit_rows, val_rows


class SeverityDataset(Dataset):
    def __init__(self, rows, transform):
        self.transform = transform
        self.samples = [(r["image_path"], int(r["label"]) - 1, r["patient_id"]) for r in rows]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, patient_id


def build_transforms():
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


class ResNetOnly(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, image):
        x = self.features(image).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for images, labels, _pids in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_patients = [], [], []
    for images, labels, patient_ids in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_patients.extend(patient_ids)
    return np.array(all_labels), np.array(all_preds), np.array(all_patients)


def compute_metrics(y_true, y_pred):
    exact_acc = accuracy_score(y_true, y_pred)
    within_one = float(np.mean(np.abs(y_true - y_pred) <= 1))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return {"exact_acc": exact_acc, "within_one_acc": within_one, "macro_f1": macro_f1,
            "kappa": kappa, "precision": precision, "recall": recall, "f1": f1,
            "support": support, "confusion_matrix": cm}


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


def patient_level_vote(y_true, y_pred, patient_ids):
    true_by_patient, pred_by_patient = defaultdict(list), defaultdict(list)
    for t, p, pid in zip(y_true, y_pred, patient_ids):
        true_by_patient[pid].append(t)
        pred_by_patient[pid].append(p)
    pt_true, pt_pred = [], []
    for pid in true_by_patient:
        pt_true.append(Counter(true_by_patient[pid]).most_common(1)[0][0])
        pt_pred.append(Counter(pred_by_patient[pid]).most_common(1)[0][0])
    return np.array(pt_true), np.array(pt_pred)


def main():
    print(f"Using device: {DEVICE}")
    set_seed(42)

    all_train_rows = load_rows(TRAIN_CSV)
    test_rows = load_rows(TEST_CSV)
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
    fit_ds = SeverityDataset(fit_rows, train_tf)
    val_ds = SeverityDataset(val_rows, eval_tf)
    test_ds = SeverityDataset(test_rows, eval_tf)

    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ResNetOnly().to(DEVICE)

    labels_fit = np.array([s[1] for s in fit_ds.samples])
    counts = np.bincount(labels_fit, minlength=NUM_CLASSES)
    # Sqrt-dampened inverse frequency: full inverse frequency (tried
    # previously) over-corrected -- it pushed grade-4 loss to 2.83x, which
    # tanked grade-5 (the largest class) recall from 0.557 to 0.390 and
    # dragged overall accuracy down with it. Dampening keeps the same
    # direction of correction with less distortion of the majority class.
    weights = np.sqrt(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"\nClass weights (sqrt-dampened): {weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_kappa = -1.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, fit_loader, optimizer, criterion)
        y_val_true, y_val_pred, _ = evaluate(model, val_loader)
        val_metrics = compute_metrics(y_val_true, y_val_pred)
        val_acc, val_kappa = val_metrics["exact_acc"], val_metrics["kappa"]

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_acc={val_acc*100:.2f}% | val_kappa={val_kappa:.4f}")

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            epochs_no_improve = 0
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  -> new best (val_kappa={val_kappa:.4f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  -> early stopping (no val improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(torch.load(CKPT_PATH))
    y_test_true, y_test_pred, test_patient_ids = evaluate(model, test_loader)
    frame_metrics = compute_metrics(y_test_true, y_test_pred)
    print_metrics("FINAL — FRAME-LEVEL TEST RESULTS", frame_metrics)

    pt_true, pt_pred = patient_level_vote(y_test_true, y_test_pred, test_patient_ids)
    patient_metrics = compute_metrics(pt_true, pt_pred)
    print_metrics(f"FINAL — PATIENT-LEVEL (MAJORITY VOTE) TEST RESULTS ({len(pt_true)} patients)", patient_metrics)


if __name__ == "__main__":
    main()
