"""
Check grade distribution (1-5) in train.csv and test.csv,
including per-grade patient counts.
"""

import csv
from collections import defaultdict

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"


def analyze(csv_path):
    """
    Returns:
        grade_counts: dict {grade: frame_count}
        grade_patients: dict {grade: set of patient_ids}
    """

    grade_counts = defaultdict(int)
    grade_patients = defaultdict(set)
    total = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = row["label"]

            if label == "no_face":
                continue

            grade_counts[label] += 1
            grade_patients[label].add(row["patient_id"])
            total += 1

    return grade_counts, grade_patients, total


def print_distribution(name, grade_counts, grade_patients, total):

    print("=" * 60)
    print(f"{name} DISTRIBUTION")
    print("=" * 60)

    for grade in [str(g) for g in range(1, 6)]:
        count = grade_counts.get(grade, 0)
        n_patients = len(grade_patients.get(grade, set()))
        pct = (100 * count / total) if total > 0 else 0
        print(
            f"Grade {grade}: {count:>6,} frames  "
            f"({pct:5.1f}%)   {n_patients} patients"
        )

    print("-" * 60)
    print(f"Total: {total:,} frames")
    print()


def main():

    train_counts, train_patients, train_total = analyze(TRAIN_CSV)
    test_counts, test_patients, test_total = analyze(TEST_CSV)

    print_distribution("TRAIN", train_counts, train_patients, train_total)
    print_distribution("TEST", test_counts, test_patients, test_total)

    # ----------------------------------------------------------
    # Combined side-by-side comparison
    # ----------------------------------------------------------

    print("=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"{'Grade':<8}{'Train':<12}{'Test':<12}{'Train %':<10}{'Test %':<10}")

    for grade in [str(g) for g in range(1, 6)]:
        tr = train_counts.get(grade, 0)
        te = test_counts.get(grade, 0)
        tr_pct = (100 * tr / train_total) if train_total > 0 else 0
        te_pct = (100 * te / test_total) if test_total > 0 else 0

        print(
            f"{grade:<8}{tr:<12,}{te:<12,}"
            f"{tr_pct:<10.1f}{te_pct:<10.1f}"
        )

    print("-" * 60)
    print(f"{'Total':<8}{train_total:<12,}{test_total:<12,}")

    # ----------------------------------------------------------
    # Patient-leakage sanity check
    # ----------------------------------------------------------

    all_train_patients = set()
    all_test_patients = set()

    for grade in train_patients:
        all_train_patients |= train_patients[grade]

    for grade in test_patients:
        all_test_patients |= test_patients[grade]

    overlap = all_train_patients & all_test_patients

    print("\n" + "=" * 60)
    print("PATIENT LEAKAGE CHECK")
    print("=" * 60)

    if overlap:
        print(f"WARNING: {len(overlap)} patients appear in BOTH train and test:")
        print(sorted(overlap))
    else:
        print("OK: No patient appears in both train and test.")


if __name__ == "__main__":
    main()