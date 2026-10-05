"""
Builds a filtered, patient-wise stratified train/test split from
auto_labels.csv.

Filtering: drops the bottom quartile of frames by anchor-prototype
margin (the ones closest to a tie between two adjacent severity
grades) -- these are the most label-noisy frames. The threshold is
picked from the actual margin distribution rather than the unvalidated
fixed constants in labeling.py.

Splitting: same patient-wise, dominant-grade-stratified logic as
split_train_test.py (no patient appears in both train and test).
"""

import csv
import random
from collections import defaultdict

INPUT_CSV = "auto_labels.csv"
TRAIN_CSV = "train_clean.csv"
TEST_CSV = "test_clean.csv"

TRAIN_FRACTION = 0.80
RANDOM_SEED = 42
MARGIN_KEEP_PERCENTILE = 25  # drop bottom 25% by margin


def main():
    random.seed(RANDOM_SEED)

    rows = []
    with open(INPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["label"] == "no_face":
                continue
            rows.append(row)

    print(f"Loaded {len(rows):,} labeled frames.")

    margins = sorted(float(r["margin"]) for r in rows)
    idx = int(len(margins) * MARGIN_KEEP_PERCENTILE / 100)
    margin_threshold = margins[idx]
    print(f"Margin p{MARGIN_KEEP_PERCENTILE} threshold: {margin_threshold:.4f}")

    rows = [r for r in rows if float(r["margin"]) >= margin_threshold]
    print(f"After margin filter: {len(rows):,} frames kept.")

    patient_label_counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        patient_label_counts[row["patient_id"]][row["label"]] += 1

    patient_dominant_label = {
        pid: max(counts, key=counts.get)
        for pid, counts in patient_label_counts.items()
    }

    patients_by_grade = defaultdict(list)
    for pid, dom in patient_dominant_label.items():
        patients_by_grade[dom].append(pid)

    train_patients = set()
    test_patients = set()

    print("\nPATIENT-LEVEL SPLIT PER GRADE")
    for grade in sorted(patients_by_grade, key=lambda x: int(x)):
        patient_list = patients_by_grade[grade]
        random.shuffle(patient_list)
        n_patients = len(patient_list)
        n_train = round(n_patients * TRAIN_FRACTION)
        if n_patients > 1 and n_train == n_patients:
            n_train -= 1
        train_ids = patient_list[:n_train]
        test_ids = patient_list[n_train:]
        train_patients.update(train_ids)
        test_patients.update(test_ids)
        print(f"Grade {grade}: {n_patients} patients -> {len(train_ids)} train / {len(test_ids)} test")

    overlap = train_patients & test_patients
    assert len(overlap) == 0, f"Patient leakage detected: {overlap}"

    train_rows = [r for r in rows if r["patient_id"] in train_patients]
    test_rows = [r for r in rows if r["patient_id"] in test_patients]

    with open(TRAIN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(TEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"\nTrain frames: {len(train_rows):,} ({len(train_patients)} patients)")
    print(f"Test frames : {len(test_rows):,} ({len(test_patients)} patients)")

    def grade_dist(row_list):
        c = defaultdict(int)
        for r in row_list:
            c[r["label"]] += 1
        return c

    train_dist = grade_dist(train_rows)
    test_dist = grade_dist(test_rows)
    total_train = sum(train_dist.values())
    total_test = sum(test_dist.values())

    print("\nGrade distribution (train% / test%):")
    for grade in sorted(set(train_dist) | set(test_dist), key=lambda x: int(x)):
        tp = 100 * train_dist.get(grade, 0) / total_train
        te = 100 * test_dist.get(grade, 0) / total_test
        print(f"  Grade {grade}: train={train_dist.get(grade,0):,} ({tp:.1f}%)  test={test_dist.get(grade,0):,} ({te:.1f}%)")

    print(f"\nWrote {TRAIN_CSV}, {TEST_CSV}")


if __name__ == "__main__":
    main()
