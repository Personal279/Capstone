"""
Facial Palsy Auto Labeling Script
Uses dlib 68-point landmarks to compute facial asymmetry.

FAST version: skips face detection entirely (frames are already 300x300
face crops, so the whole image is passed directly to the landmark predictor).
Runs at ~200-500 img/s on CPU vs 0.7 img/s with face_alignment.

Setup (one time):
    pip install dlib opencv-python pandas
    # Download the landmark model (~100 MB):
    curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
         -o shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2

Run:
    python3 labeling.py
    python3 labeling.py /path/to/frames
"""

import cv2
import dlib
import numpy as np
import pandas as pd
import sys, time, os
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import dlib
except ImportError:
    print("Missing dependency. Run:  pip install dlib")
    exit(1)

WORKERS       = max(1, (os.cpu_count() or 4) - 1)
PREDICTOR_PATH = str(Path(__file__).parent / "shape_predictor_68_face_landmarks.dat")

if not Path(PREDICTOR_PATH).exists():
    print("Landmark model not found. Run:")
    print("  curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -o shape_predictor_68_face_landmarks.dat.bz2")
    print("  bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
    exit(1)

# ── 68-point landmark layout ──────────────────────────────────────────────────
# Left eye:   36-41   Right eye:   42-47
# Left brow:  17-21   Right brow:  22-26
# Nose tip:   30      Nose base:   33
# Mouth L:    48      Mouth R:     54
# Jaw:        0-16    Chin:        8


# ── Per-worker init ───────────────────────────────────────────────────────────
_predictor = None

def _init_worker():
    global _predictor
    _predictor = dlib.shape_predictor(PREDICTOR_PATH)


# ── Geometry ──────────────────────────────────────────────────────────────────
def dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_opening(lms, indices):
    top    = (lms[indices[1]] + lms[indices[2]]) / 2
    bottom = (lms[indices[5]] + lms[indices[4]]) / 2
    return dist(top, bottom)


def compute_asymmetry(lms):
    scores = {}

    l_eye = eye_opening(lms, [36, 37, 38, 39, 40, 41])
    r_eye = eye_opening(lms, [42, 43, 44, 45, 46, 47])
    scores["eye"] = abs(l_eye - r_eye) / max(l_eye, r_eye, 1e-5) * 100

    nose_tip = lms[30]
    l_brow   = np.mean(lms[17:22], axis=0)
    r_brow   = np.mean(lms[22:27], axis=0)
    scores["eyebrow"] = abs(dist(nose_tip, l_brow) - dist(nose_tip, r_brow)) / \
                        max(dist(nose_tip, l_brow), dist(nose_tip, r_brow), 1e-5) * 100

    nose_base = lms[33]
    l_mouth   = dist(nose_base, lms[48])
    r_mouth   = dist(nose_base, lms[54])
    scores["mouth"] = abs(l_mouth - r_mouth) / max(l_mouth, r_mouth, 1e-5) * 100

    l_cheek = dist(nose_base, lms[3])
    r_cheek = dist(nose_base, lms[13])
    scores["cheek"] = abs(l_cheek - r_cheek) / max(l_cheek, r_cheek, 1e-5) * 100

    chin  = lms[8]
    l_jaw = dist(chin, lms[2])
    r_jaw = dist(chin, lms[14])
    scores["jaw"] = abs(l_jaw - r_jaw) / max(l_jaw, r_jaw, 1e-5) * 100

    weights = {"eye": 0.30, "mouth": 0.30, "eyebrow": 0.20, "cheek": 0.10, "jaw": 0.10}
    overall = sum(scores[k] * weights[k] for k in weights)
    return round(overall, 2), {k: round(v, 2) for k, v in scores.items()}


def asymmetry_to_grade(score):
    if score < 8:    return 1
    elif score < 18: return 2
    elif score < 35: return 3
    elif score < 55: return 4
    else:            return 5


# ── Worker function ───────────────────────────────────────────────────────────
def _process_one(image_path_str):
    image_bgr = cv2.imread(image_path_str)
    if image_bgr is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Skip detection: pass whole frame as the face bounding box
    rect  = dlib.rectangle(0, 0, w, h)
    shape = _predictor(gray, rect)
    lms   = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)],
                     dtype=np.float32)
    overall_score, region_scores = compute_asymmetry(lms)
    grade = asymmetry_to_grade(overall_score)
    return image_path_str, grade, overall_score, region_scores


# ── Main ──────────────────────────────────────────────────────────────────────
def label_dataset(dataset_root, output_csv="labels.csv"):
    dataset_root = Path(dataset_root)
    records      = []
    supported    = {".jpg", ".jpeg", ".png", ".bmp"}

    print(f"Using {WORKERS} parallel workers\n")

    for split in ["train", "val", "test"]:
        split_path = dataset_root / split
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping.")
            continue

        # NORMAL patients (in paralysed folder due to swap) -> Grade 1
        normal_path = split_path / "paralysed"
        if normal_path.exists():
            normal_images = [f for f in normal_path.rglob("*")
                             if f.suffix.lower() in supported]
            print(f"{split}/paralysed (normal patients) -- {len(normal_images)} images -> Grade 1")
            for img_path in normal_images:
                records.append({
                    "image_path": str(img_path), "grade": 1,
                    "asymmetry_score": 0.0, "eye_score": 0.0,
                    "mouth_score": 0.0, "eyebrow_score": 0.0,
                    "cheek_score": 0.0, "jaw_score": 0.0,
                    "split": split, "patient_id": "normal",
                })

        # PARALYSED patients (in normal folder due to swap) -> grading
        paralysed_path = split_path / "normal"
        if not paralysed_path.exists():
            continue

        patient_folders = [f for f in paralysed_path.iterdir() if f.is_dir()]
        if not patient_folders:
            patient_folders = [paralysed_path]

        for patient_folder in sorted(patient_folders):
            patient_id = patient_folder.name
            images     = [str(f) for f in patient_folder.rglob("*")
                          if f.suffix.lower() in supported]
            total      = len(images)
            print(f"\n{split}/normal (paralysed patient {patient_id}) -- {total} images")

            success = fail = done = 0
            start       = time.time()
            result_map  = {}

            with ProcessPoolExecutor(max_workers=WORKERS,
                                     initializer=_init_worker) as pool:
                futures = {pool.submit(_process_one, p): p for p in images}
                for fut in as_completed(futures):
                    done += 1
                    result = fut.result()
                    if result is None:
                        fail += 1
                    else:
                        success += 1
                        result_map[result[0]] = result[1:]

                    elapsed  = time.time() - start
                    rate     = done / max(elapsed, 1e-5)
                    eta      = int((total - done) / rate) if rate > 0 else 0
                    bar_fill = int(30 * done / total)
                    bar      = "#" * bar_fill + "-" * (30 - bar_fill)
                    print(f"  [{bar}] {done}/{total}  ok={success} fail={fail}"
                          f"  {rate:.1f} img/s  ETA {timedelta(seconds=eta)}",
                          end="\r", flush=True)

            print()
            print(f"   Labeled: {success}  No face detected: {fail}")

            for img_path in images:
                if img_path in result_map:
                    grade, overall, regions = result_map[img_path]
                    records.append({
                        "image_path": img_path, "grade": grade,
                        "asymmetry_score": overall,
                        "eye_score":       regions["eye"],
                        "mouth_score":     regions["mouth"],
                        "eyebrow_score":   regions["eyebrow"],
                        "cheek_score":     regions["cheek"],
                        "jaw_score":       regions["jaw"],
                        "split":           split,
                        "patient_id":      patient_id,
                    })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print("\n" + "="*50)
    print("LABELING COMPLETE")
    print("="*50)
    print(f"Total images labeled : {len(df)}")
    print(f"Output CSV           : {Path(output_csv).absolute()}")
    print("\nGrade distribution:")
    grade_names = {1: "Normal", 2: "Mild", 3: "Moderate", 4: "Severe", 5: "Total paralysis"}
    for grade, count in df["grade"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"  Grade {grade} ({grade_names[grade]:16s}): {count:4d} images ({pct:.1f}%)")
    print("\nSplit distribution:")
    for split, count in df["split"].value_counts().items():
        print(f"  {split}: {count} images")
    return df


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
                   str(Path(__file__).parent / "palsynet_frames" / "frames")
    print(f"Dataset : {dataset_path}")
    print("Starting...\n")
    label_dataset(dataset_path, "labels.csv")