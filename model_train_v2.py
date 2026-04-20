"""
model_train_v2.py  —  House-Brackmann Grading  |  Capstone Batch 201
======================================================================
Target: Beat v1's 72.8% by fixing the Grade 2 ↔ 3 confusion.

Key changes vs v1:
  - Richer FAI: 10-dim asymmetry vector (vs 1-dim in v1)
  - Augments ANY underrepresented grade automatically (not just 4-5)
  - Trains MLP, SVM, Random Forest, Gradient Boosting
  - Prints accuracy + report + confusion matrix for all models
  - Grade 2/3 boundary analysis printed separately at the end
  - Ready for Grade 4/5 data later — no hardcoded grade assumptions

Run:
    python model_train_v2.py
    python model_train_v2.py --csv path/to/labels_fixed.csv
"""

import os
import argparse
import warnings
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T

import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CSV_PATH = r"C:\Users\shreya\Desktop\Capstone\Capstone-master\labels_yfp.csv"
SAVE_DIR = r"C:\Users\shreya\Desktop\Capstone\Capstone-master\saved_models_v2"

# Augmentation: any grade with fewer samples than AUG_RATIO × majority class
# will be boosted automatically — works for any grade mix (1-3 now, 4-5 later)
AUG_RATIO      = 0.75   # target = 75% of majority class count
AUG_MIN_NEEDED = 10     # skip if gap is tiny

# v1 baseline to compare against
V1_ACCURACY = 0.728

# =============================================================================
# SETUP
# =============================================================================
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 62)
print("  HB Grading — model_train_v2")
print("=" * 62)
print(f"  Device : {device}")
print(f"  CSV    : {CSV_PATH}")
print(f"  Save   : {SAVE_DIR}")

# =============================================================================
# MEDIAPIPE
# =============================================================================
_mp_face   = mp.solutions.face_mesh
_face_mesh = _mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True)
# =============================================================================
# FAI — 10-dim asymmetry vector
# =============================================================================
# Why 10-dim instead of v1's 1-dim:
#   v1 collapsed everything into one weighted scalar, losing the regional
#   breakdown that separates Grade 2 from Grade 3.
#   Grade 2 (mild) typically shows brow/forehead asymmetry first;
#   Grade 3 (moderate) adds eye and lip involvement.
#   Individual region scores + interaction terms let the classifier
#   learn that boundary directly.

def _compute_fai_vector(lm) -> np.ndarray:
    eye_L    = abs(lm[159].y - lm[145].y)          # left eye openness
    eye_R    = abs(lm[386].y - lm[374].y)          # right eye openness
    eye_asym = abs(eye_L - eye_R) / (max(eye_L, eye_R) + 1e-6)
    lip      = abs(lm[61].y  - lm[291].y)
    brow     = abs(lm[70].y  - lm[300].y)
    cheek    = abs(lm[50].y  - lm[280].y)
    jaw      = abs(lm[152].y - lm[10].y)

    fai_weighted      = 0.30*eye_asym + 0.30*lip + 0.15*brow + 0.15*cheek + 0.10*jaw
    eye_brow_combined = eye_asym * brow   # upper face interaction — key for G2 vs G3
    lip_jaw_combined  = lip * jaw         # lower face interaction

    return np.array([
        fai_weighted,
        eye_L, eye_R, eye_asym,
        lip, brow, cheek, jaw,
        eye_brow_combined,
        lip_jaw_combined,
    ], dtype=np.float32)


def extract_fai(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(10, dtype=np.float32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = _face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        return np.zeros(10, dtype=np.float32)
    return _compute_fai_vector(result.multi_face_landmarks[0].landmark)


def _fai_from_pil(img_pil: Image.Image) -> np.ndarray:
    img_np = np.array(img_pil.convert("RGB"))
    result = _face_mesh.process(img_np)
    if not result.multi_face_landmarks:
        return np.zeros(10, dtype=np.float32)
    return _compute_fai_vector(result.multi_face_landmarks[0].landmark)


# =============================================================================
# RESNET50
# =============================================================================
_weights   = ResNet50_Weights.DEFAULT
_resnet    = resnet50(weights=_weights)
_resnet    = torch.nn.Sequential(*list(_resnet.children())[:-1])
_resnet    = _resnet.to(device).eval()
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_weights.transforms().mean,
                std=_weights.transforms().std),
])

def _resnet_feat(img_pil: Image.Image) -> np.ndarray:
    tensor = _transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = _resnet(tensor)
    return feat.cpu().flatten().numpy()

def extract_resnet(image_path: str) -> np.ndarray:
    try:
        return _resnet_feat(Image.open(image_path).convert("RGB"))
    except Exception:
        return np.zeros(2048, dtype=np.float32)


# =============================================================================
# FEATURE FUSION  (2058-dim total)
# =============================================================================
FAI_WEIGHT = 20.0   # higher than v1's 10× — FAI is now 10-dim so scale matters

def fuse(fai: np.ndarray, res: np.ndarray) -> np.ndarray:
    return np.concatenate([fai * FAI_WEIGHT, res])


# =============================================================================
# AUGMENTATION
# =============================================================================
_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=12),
    T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
    T.RandomResizedCrop(224, scale=(0.88, 1.00)),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

def augment_grade(paths: list, grade: int, n_needed: int):
    print(f"   ↑ Grade {grade}: generating +{n_needed} samples "
          f"from {len(paths)} source images...")
    X_aug, y_aug = [], []
    count = 0
    while count < n_needed:
        for p in paths:
            if count >= n_needed:
                break
            try:
                aug  = _aug(Image.open(p).convert("RGB"))
                feat = fuse(_fai_from_pil(aug), _resnet_feat(aug))
                X_aug.append(feat)
                y_aug.append(grade)
                count += 1
            except Exception:
                pass
    return np.array(X_aug), np.array(y_aug)


# =============================================================================
# CSV LOADING
# =============================================================================
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    path_col  = next((c for c in df.columns if "image" in c or "path" in c), df.columns[0])
    grade_col = next((c for c in df.columns if "grade" in c or "label" in c), df.columns[1])
    df = df.rename(columns={path_col: "image_path", grade_col: "grade"})
    df["grade"] = df["grade"].astype(str).str.strip().str.extract(r'(\d+)').astype(int)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    grade_names = {1:"Normal", 2:"Mild", 3:"Moderate", 4:"Severe", 5:"Total palsy"}
    counts = df["grade"].value_counts().sort_index()
    max_c  = counts.max()
    print(f"\n📂 CSV loaded: {len(df)} valid images")
    print("   Grade distribution (original):")
    for g, c in counts.items():
        bar = "█" * int(c / max_c * 24)
        print(f"   Grade {g} ({grade_names.get(g,'?'):12s}): {c:4d}  {bar}")
    return df


# =============================================================================
# MAIN
# =============================================================================
def main(csv_path: str):
    df = load_csv(csv_path)

    # ── Feature extraction ────────────────────────────────────────────────────
    print("\n🔄 Extracting features...")
    X, y, paths_by_grade = [], [], {}
    for i, row in df.iterrows():
        path  = row["image_path"]
        grade = int(row["grade"])
        feat  = fuse(extract_fai(path), extract_resnet(path))
        X.append(feat)
        y.append(grade)
        paths_by_grade.setdefault(grade, []).append(path)
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"   ✔ {i+1}/{len(df)}", end="\r")

    X = np.array(X)
    y = np.array(y)
    print(f"\n   ✅ Done — feature dim: {X.shape[1]}")

    # ── Auto-augmentation (any underrepresented grade) ────────────────────────
    counts   = {g: int((y == g).sum()) for g in np.unique(y)}
    majority = max(counts.values())
    target   = int(majority * AUG_RATIO)

    print(f"\n🔀 Auto-augmentation (target per grade ≥ {target}):")
    X_list, y_list = list(X), list(y)

    for grade, current in sorted(counts.items()):
        n_needed = target - current
        if n_needed >= AUG_MIN_NEEDED and grade in paths_by_grade:
            Xa, ya = augment_grade(paths_by_grade[grade], grade, n_needed)
            X_list.extend(Xa)
            y_list.extend(ya)
        else:
            reason = "already sufficient" if n_needed < AUG_MIN_NEEDED else "no source images"
            print(f"   – Grade {grade}: {current} samples — {reason}, skipping")

    X_all = np.array(X_list)
    y_all = np.array(y_list)

    grade_names  = {1:"Normal", 2:"Mild", 3:"Moderate", 4:"Severe", 5:"Total palsy"}
    counts_aug   = {g: int((y_all == g).sum()) for g in np.unique(y_all)}
    max_c        = max(counts_aug.values())
    print("\n   Grade distribution (after augmentation):")
    for g, c in sorted(counts_aug.items()):
        bar = "█" * int(c / max_c * 24)
        print(f"   Grade {g} ({grade_names.get(g,'?'):12s}): {c:4d}  {bar}")

    # ── Normalise + split ─────────────────────────────────────────────────────
    print("\n⚖️  Normalising features...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_v2.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    present_grades = sorted(np.unique(y_all))
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Model definitions ─────────────────────────────────────────────────────
    models = {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256),
            activation="relu",
            learning_rate_init=0.0005,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
    }

    # ── Train + evaluate ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  TRAINING & EVALUATION")
    print("=" * 62)

    results   = {}
    best_name = None
    best_acc  = -1.0
    tnames    = [f"G{g}-{grade_names.get(g,'?')[:3]}" for g in present_grades]

    for name, clf in models.items():
        print(f"\n🚀 Training: {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        results[name] = {"model": clf, "acc": acc, "pred": y_pred}

        beat = "↑ BEATS v1" if acc > V1_ACCURACY else f"↓ below v1 ({V1_ACCURACY*100:.1f}%)"
        print(f"\n{'─'*62}")
        print(f"  {name}")
        print(f"{'─'*62}")
        print(f"  🎯 Accuracy : {acc*100:.2f}%  ({beat})")
        print(f"\n  Per-class report:")
        print(classification_report(y_test, y_pred,
                                    labels=present_grades,
                                    target_names=tnames,
                                    zero_division=0))
        print(f"  Confusion matrix (rows=true  cols=pred):")
        cm     = confusion_matrix(y_test, y_pred, labels=present_grades)
        header = "         " + "  ".join(f"G{g}" for g in present_grades)
        print(header)
        for i, g in enumerate(present_grades):
            row = "  ".join(f"{v:4d}" for v in cm[i])
            print(f"  G{g} true | {row}")

        if acc > best_acc:
            best_acc  = acc
            best_name = name

        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(clf, os.path.join(SAVE_DIR, f"{safe}.pkl"))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  {'Model':<25}  {'Accuracy':>10}  {'vs v1 (72.8%)'}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*14}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        delta  = (r["acc"] - V1_ACCURACY) * 100
        marker = "  ← BEST" if name == best_name else ""
        sign   = "+" if delta >= 0 else ""
        print(f"  {name:<25}  {r['acc']*100:>9.2f}%  {sign}{delta:+.1f}%{marker}")

    print(f"\n  🏆 Best model : {best_name}  ({best_acc*100:.2f}%)")
    print(f"     Saved to   : {SAVE_DIR}")

    # ── Grade 2 ↔ 3 boundary breakdown ───────────────────────────────────────
    if 2 in present_grades and 3 in present_grades:
        print("\n" + "=" * 62)
        print("  GRADE 2 ↔ 3 BOUNDARY  (main weakness in v1)")
        print("=" * 62)
        print(f"  {'Model':<25}  {'G2 recall':>10}  {'G3 recall':>10}  {'G2→G3 errors':>14}")
        print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*14}")
        for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
            pred      = r["pred"]
            g2_mask   = y_test == 2
            g3_mask   = y_test == 3
            g2_recall = (pred[g2_mask] == 2).mean() * 100 if g2_mask.any() else float("nan")
            g3_recall = (pred[g3_mask] == 3).mean() * 100 if g3_mask.any() else float("nan")
            g2_to_g3  = int(((pred == 3) & g2_mask).sum())
            print(f"  {name:<25}  {g2_recall:>9.1f}%  {g3_recall:>9.1f}%  {g2_to_g3:>14}")
        print(f"\n  v1 reference    :  {'48.0%':>10}  {'68.0%':>10}  {'29':>14}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    _face_mesh.close()
    print(f"\n✅ DONE\n")


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    args = parser.parse_args()
    main(args.csv)
