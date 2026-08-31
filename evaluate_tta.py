"""
Test-time augmentation + soft-vote patient aggregation on the PROVEN
Experiment A checkpoint. No retraining -- this only changes how
predictions are extracted from the already-selected best model, so it
carries none of the overfitting risk that hit every training-time
modification tried today (FAI-aux, class weights, dampened weights).

TTA views: original + mild rotations/brightness jitter (matching the
train-time augmentation ranges). Deliberately NO horizontal flip --
that would mirror which side is paralyzed and corrupt the asymmetry
signal the model relies on.

Softmax probabilities are averaged across views (not just each view's
hard prediction), then argmax'd -- and the same probability-averaging
is used again across a patient's frames for patient-level aggregation,
which uses more information than hard majority voting.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score,
)

from Model_train import load_rows, TEST_CSV, NUM_CLASSES, IMAGE_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ResNetOnly(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, image):
        x = self.features(image).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation((5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation((-5, -5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]),
]


class MultiViewDataset(Dataset):
    def __init__(self, rows):
        self.samples = [(r["image_path"], int(r["label"]) - 1, r["patient_id"]) for r in rows]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        views = torch.stack([tf(image) for tf in TTA_TRANSFORMS])  # (V, C, H, W)
        return views, label, patient_id


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


def main():
    print(f"Using device: {DEVICE}")
    test_rows = load_rows(TEST_CSV)
    test_ds = MultiViewDataset(test_rows)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)
    print(f"[test] {len(test_ds)} images, {len(TTA_TRANSFORMS)} TTA views each")

    model = ResNetOnly().to(DEVICE)
    model.load_state_dict(torch.load("best_model_experiment_A.pt", map_location=DEVICE))
    model.eval()

    all_labels, all_patients = [], []
    all_probs = []  # averaged-over-views softmax prob per frame

    with torch.no_grad():
        for views, labels, patient_ids in test_loader:
            B, V, C, H, W = views.shape
            flat = views.view(B * V, C, H, W).to(DEVICE)
            logits = model(flat)
            probs = F.softmax(logits, dim=1).view(B, V, NUM_CLASSES).mean(dim=1)  # avg over views
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_patients.extend(patient_ids)

    all_probs = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_labels)
    y_pred_tta = all_probs.argmax(axis=1)

    frame_metrics = compute_metrics(y_true, y_pred_tta)
    print_metrics("FRAME-LEVEL TEST RESULTS (TTA, 4-view soft average)", frame_metrics)

    # Patient-level: average probabilities across all of a patient's frames
    # (soft vote), not just the mode of hard per-frame predictions.
    prob_by_patient = defaultdict(list)
    label_by_patient = defaultdict(list)
    for probs, t, pid in zip(all_probs, y_true, all_patients):
        prob_by_patient[pid].append(probs)
        label_by_patient[pid].append(t)

    from collections import Counter
    pt_true, pt_pred = [], []
    for pid in prob_by_patient:
        avg_prob = np.mean(prob_by_patient[pid], axis=0)
        pt_pred.append(int(avg_prob.argmax()))
        pt_true.append(Counter(label_by_patient[pid]).most_common(1)[0][0])

    patient_metrics = compute_metrics(np.array(pt_true), np.array(pt_pred))
    print_metrics(
        f"PATIENT-LEVEL (SOFT-VOTE, PROBABILITY AVERAGED) TEST RESULTS ({len(pt_true)} patients)",
        patient_metrics,
    )


if __name__ == "__main__":
    main()
