"""
Automated anchor-based severity pseudo-labeling
for facial paralysis images.

Pipeline:

    Image
      |
      v
    MediaPipe Face Mesh
      |
      v
    Facial asymmetry features
      |
      v
    Normalize feature scales
      |
      v
    Compare with severity anchor PROTOTYPES
      (average of multiple anchor images per grade)
      |
      v
    Nearest-prototype pseudo-label (1-5)

IMPORTANT:
These are pseudo-labels, NOT clinically confirmed ground-truth labels.

Output:
    auto_labels.csv

Columns include:
    image_path
    original_split
    condition_folder
    patient_id
    label
    confidence
    margin
    low_margin_flag
    no_face_detected
    is_anchor_patient

The original split is retained only as metadata.
We will NOT use it for the final train/val/test split.
The next step should be a patient-wise split.
"""

import os

# Must be set before importing cv2/mediapipe -- on this many-core ARM box,
# the default per-library thread pools (OpenCV, TFLite/XNNPACK, BLAS) can
# oversubscribe the CPU and intermittently deadlock in futex waits during
# a long single-process run over thousands of images.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import cv2
cv2.setNumThreads(1)
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"/home/student-8/Capstone/palsynet_frames/frames"

OUTPUT_CSV = "auto_labels.csv"

# ------------------------------------------------------------
# ANCHOR_PATHS now maps each grade to a LIST of anchor images
# instead of a single image. All anchor images for a grade are
# averaged (centroid) into one prototype feature vector for
# that grade, which makes the labeling much less sensitive to
# any single anchor being unusual/mislabeled.
#
# Prefer DIFFERENT patients within each grade's list where
# possible, for a more representative prototype.
# ------------------------------------------------------------

ANCHOR_PATHS = {

    1: [
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/2/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/3/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/4/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/5/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/7/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/9/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/10/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/11/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/12/frame_00001.jpg",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/paralysed/13/frame_00001.jpg",
    ],

    2: [
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y17/401.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y16/911.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y1/1361.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y28/10761.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y9/3511.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y10/466.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y14/12331.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y17/4286.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y26/2046.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y11/9536.bmp",
    ],

    3: [
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/911.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y23/101.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y17/6166.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y15/9151.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y26/911.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y27/5056.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y11/3841.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y8/21316.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y4/911.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y3/3276.bmp",
    ],

    4: [
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/901.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y17/1691.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y15/6816.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y26/941.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y27/4521.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y8/4506.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y3/1336.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y16/896.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y1/4066.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y28/1691.bmp",
    ],

    5: [
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/1691.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y15/10176.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y26/1071.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y1/11386.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y14/6056.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/426.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/3566.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/3376.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/1646.bmp",
        r"/home/student-8/Capstone/palsynet_frames/frames/train/normal/y22/1656.bmp",
    ],
}

NUM_CLASSES = 5

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

# ------------------------------------------------------------
# Ambiguity thresholds
# ------------------------------------------------------------

LOW_MARGIN_THRESHOLD = 0.10
LOW_CONFIDENCE_THRESHOLD = 0.55

# ------------------------------------------------------------
# Feature weights
# ------------------------------------------------------------

FEATURE_WEIGHTS = np.array([
    1.50,   # eye asymmetry
    1.25,   # eyebrow asymmetry
    1.50,   # mouth asymmetry
    1.25,   # mouth vertical asymmetry
    0.50,   # mouth openness
], dtype=np.float32)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_BROW = 105
RIGHT_BROW = 334

NOSE_BRIDGE = 6
CHIN = 152

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

MOUTH_TOP = 13
MOUTH_BOTTOM = 14


# ============================================================
# HELPERS
# ============================================================

def cross2d(a, b):
    """
    Scalar 2D cross product.

    NumPy 2.0 removed support for 2D vectors in np.cross(),
    which raises a ValueError. This is a drop-in replacement
    for the specific 2D case used in this script.
    """
    return a[0] * b[1] - a[1] * b[0]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):
    """
    Extract five facial asymmetry features.

    Returns:
        np.ndarray of shape (5,)
        or None if a usable face cannot be detected.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    h, w, _ = image.shape

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    def pt(index):
        return np.array(
            [
                landmarks[index].x * w,
                landmarks[index].y * h
            ],
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Inter-ocular distance
    # --------------------------------------------------------

    left_eye_outer = pt(LEFT_EYE_OUTER)
    right_eye_outer = pt(RIGHT_EYE_OUTER)

    inter_ocular = np.linalg.norm(
        left_eye_outer - right_eye_outer
    )

    if inter_ocular < 1e-6:
        return None

    # --------------------------------------------------------
    # Eye openness
    # --------------------------------------------------------

    left_eye_open = (
        np.linalg.norm(
            pt(LEFT_EYE_TOP) -
            pt(LEFT_EYE_BOTTOM)
        )
        / inter_ocular
    )

    right_eye_open = (
        np.linalg.norm(
            pt(RIGHT_EYE_TOP) -
            pt(RIGHT_EYE_BOTTOM)
        )
        / inter_ocular
    )

    eye_asymmetry = abs(
        left_eye_open -
        right_eye_open
    )

    # --------------------------------------------------------
    # Eyebrow asymmetry
    # --------------------------------------------------------

    left_brow_height = (
        pt(LEFT_BROW)[1] -
        pt(LEFT_EYE_TOP)[1]
    ) / inter_ocular

    right_brow_height = (
        pt(RIGHT_BROW)[1] -
        pt(RIGHT_EYE_TOP)[1]
    ) / inter_ocular

    brow_asymmetry = abs(
        left_brow_height -
        right_brow_height
    )

    # --------------------------------------------------------
    # Mouth asymmetry relative to facial midline
    # --------------------------------------------------------

    nose = pt(NOSE_BRIDGE)
    chin = pt(CHIN)

    midline = chin - nose

    midline_norm = np.linalg.norm(midline)

    if midline_norm < 1e-6:
        return None

    midline_unit = midline / midline_norm

    mouth_left = pt(MOUTH_LEFT)
    mouth_right = pt(MOUTH_RIGHT)

    left_dist = abs(
        cross2d(
            midline_unit,
            mouth_left - nose
        )
    )

    right_dist = abs(
        cross2d(
            midline_unit,
            mouth_right - nose
        )
    )

    mouth_asymmetry = (
        abs(left_dist - right_dist)
        / inter_ocular
    )

    # --------------------------------------------------------
    # Mouth vertical asymmetry
    # --------------------------------------------------------

    mouth_vertical_asymmetry = (
        abs(
            mouth_left[1] -
            mouth_right[1]
        )
        / inter_ocular
    )

    # --------------------------------------------------------
    # Mouth openness
    # --------------------------------------------------------

    mouth_open = (
        np.linalg.norm(
            pt(MOUTH_TOP) -
            pt(MOUTH_BOTTOM)
        )
        / inter_ocular
    )

    features = np.array([
        eye_asymmetry,
        brow_asymmetry,
        mouth_asymmetry,
        mouth_vertical_asymmetry,
        mouth_open,
    ], dtype=np.float32)

    if not np.all(np.isfinite(features)):
        return None

    return features


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(base_dir):
    """
    Recursively collect images from:

        frames/
            train/
            val/
            test/

    The original split is stored only for reference.
    """

    records = []

    for split in ["train", "val", "test"]:

        split_path = os.path.join(
            base_dir,
            split
        )

        if not os.path.isdir(split_path):
            print(
                f"WARNING: Missing split folder: "
                f"{split_path}"
            )
            continue

        for condition_folder in sorted(
            os.listdir(split_path)
        ):

            condition_path = os.path.join(
                split_path,
                condition_folder
            )

            if not os.path.isdir(condition_path):
                continue

            for patient_id in sorted(
                os.listdir(condition_path)
            ):

                patient_path = os.path.join(
                    condition_path,
                    patient_id
                )

                if not os.path.isdir(patient_path):
                    continue

                for root, _, files in os.walk(
                    patient_path
                ):

                    for filename in sorted(files):

                        if not filename.lower().endswith(
                            VALID_EXTENSIONS
                        ):
                            continue

                        image_path = os.path.join(
                            root,
                            filename
                        )

                        records.append({
                            "image_path": image_path,
                            "original_split": split,
                            "condition_folder": condition_folder,
                            "patient_id": patient_id,
                        })

    return records


# ============================================================
# CHECK ANCHORS
# ============================================================

def validate_anchors():
    """
    Verify all anchor files exist and report their patient IDs,
    per grade. Multiple anchors per grade are now supported.
    """

    print("=" * 70)
    print("ANCHOR VALIDATION")
    print("=" * 70)

    anchor_patients_by_grade = {}

    for label, paths in ANCHOR_PATHS.items():

        if len(paths) == 0:
            raise ValueError(
                f"Grade {label} has no anchor paths configured."
            )

        patients_this_grade = []

        print(f"\nGrade {label}: {len(paths)} anchor image(s)")

        for path in paths:

            if not os.path.isfile(path):

                raise FileNotFoundError(
                    f"\nAnchor for grade {label} does not exist:\n"
                    f"{path}"
                )

            # Expected structure:
            # .../<condition>/<patient>/<file>

            patient_id = os.path.basename(
                os.path.dirname(path)
            )

            patients_this_grade.append(patient_id)

            print(
                f"  patient={patient_id:<10} {path}"
            )

        anchor_patients_by_grade[label] = patients_this_grade

        unique_patients_this_grade = set(patients_this_grade)

        if len(unique_patients_this_grade) < len(patients_this_grade):
            print(
                f"  NOTE: Grade {label} reuses the same patient "
                f"for more than one anchor image. Extra images from "
                f"the same patient still help average out noise from "
                f"a single frame, but different patients give a "
                f"stronger prototype."
            )

    # --------------------------------------------------------
    # Cross-grade patient overlap check
    # (a patient anchoring two different grades is a red flag)
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CROSS-GRADE PATIENT OVERLAP CHECK")
    print("-" * 70)

    grade_patient_sets = {
        label: set(patients)
        for label, patients in anchor_patients_by_grade.items()
    }

    overlap_found = False

    labels = sorted(grade_patient_sets)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            label_a = labels[i]
            label_b = labels[j]

            overlap = (
                grade_patient_sets[label_a] &
                grade_patient_sets[label_b]
            )

            if overlap:
                overlap_found = True
                print(
                    f"WARNING: Patient(s) {sorted(overlap)} anchor "
                    f"BOTH grade {label_a} and grade {label_b}."
                )

    if not overlap_found:
        print("GOOD: No patient anchors more than one grade.")

    return anchor_patients_by_grade


# ============================================================
# BUILD ANCHOR PROTOTYPES (averaged features per grade)
# ============================================================

def build_anchor_prototypes():
    """
    Extract features for every anchor image and average them
    (per grade) into a single prototype feature vector.

    Returns:
        anchor_features: dict {label: np.ndarray shape (5,)}
            the averaged prototype per grade
    """

    print("\n" + "=" * 70)
    print("EXTRACTING ANCHOR FEATURES")
    print("=" * 70)

    anchor_features = {}

    for label, paths in ANCHOR_PATHS.items():

        print(f"\nGrade {label}:")

        per_image_features = []

        for path in paths:

            features = extract_features(path)

            if features is None:
                print(
                    f"  WARNING: No usable face detected, "
                    f"skipping this anchor:\n  {path}"
                )
                continue

            per_image_features.append(features)

            print(
                f"  {os.path.basename(path):<20} "
                f"{np.round(features, 5)}"
            )

        if len(per_image_features) == 0:
            raise RuntimeError(
                f"No usable face detected in ANY anchor image "
                f"for grade {label}. Cannot build a prototype."
            )

        prototype = np.mean(
            np.vstack(per_image_features),
            axis=0
        )

        anchor_features[label] = prototype

        print(
            f"  -> Grade {label} prototype "
            f"(avg of {len(per_image_features)} image(s)): "
            f"{np.round(prototype, 5)}"
        )

    return anchor_features


# ============================================================
# ROBUST FEATURE SCALING
# ============================================================

def build_feature_scaler(anchor_features):
    """
    Build a scale from anchor prototype ranges.
    """

    matrix = np.vstack([
        anchor_features[label]
        for label in sorted(anchor_features)
    ])

    feature_min = matrix.min(axis=0)
    feature_max = matrix.max(axis=0)

    feature_range = (
        feature_max -
        feature_min
    )

    feature_range[
        feature_range < 1e-6
    ] = 1.0

    return feature_min, feature_range


# ============================================================
# DISTANCE
# ============================================================

def calculate_distances(
    features,
    anchor_features,
    feature_min,
    feature_range
):
    """
    Calculate weighted normalized Euclidean distance
    to every grade's prototype.
    """

    normalized_features = (
        features -
        feature_min
    ) / feature_range

    distances = {}

    for label, anchor in anchor_features.items():

        normalized_anchor = (
            anchor -
            feature_min
        ) / feature_range

        difference = (
            normalized_features -
            normalized_anchor
        )

        weighted_difference = (
            difference *
            np.sqrt(FEATURE_WEIGHTS)
        )

        distance = np.linalg.norm(
            weighted_difference
        )

        distances[label] = float(distance)

    return distances


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    best_distance,
    second_distance
):
    """
    Convert nearest-vs-second-nearest separation
    into a simple confidence score.

    This is NOT a calibrated probability.
    """

    denominator = (
        second_distance +
        best_distance +
        1e-8
    )

    confidence = (
        second_distance -
        best_distance
    ) / denominator

    confidence = float(
        np.clip(confidence, 0.0, 1.0)
    )

    return confidence


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUTOMATED SEVERITY PSEUDO-LABELING (multi-anchor prototypes)")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate anchors
    # --------------------------------------------------------

    validate_anchors()

    anchor_patients = {
        label: set(patients)
        for label, patients in
        {
            lbl: [
                os.path.basename(os.path.dirname(p))
                for p in paths
            ]
            for lbl, paths in ANCHOR_PATHS.items()
        }.items()
    }

    anchor_patient_set = set()
    for patients in anchor_patients.values():
        anchor_patient_set |= patients

    # --------------------------------------------------------
    # Extract & average anchor features into prototypes
    # --------------------------------------------------------

    anchor_features = build_anchor_prototypes()

    # --------------------------------------------------------
    # Build feature scaler
    # --------------------------------------------------------

    feature_min, feature_range = (
        build_feature_scaler(
            anchor_features
        )
    )

    print("\nFeature minimum:")
    print(np.round(feature_min, 5))

    print("\nFeature range:")
    print(np.round(feature_range, 5))

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COLLECTING IMAGES")
    print("=" * 70)

    records = collect_images(BASE_DIR)

    print(
        f"Found {len(records):,} images."
    )

    if len(records) == 0:

        raise RuntimeError(
            "No images found. Check BASE_DIR."
        )

    # --------------------------------------------------------
    # Label images
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABELING")
    print("=" * 70)

    fieldnames = [
        "image_path",
        "original_split",
        "condition_folder",
        "patient_id",
        "label",

        "confidence",
        "distance_to_assigned_anchor",

        "second_closest_label",
        "second_closest_distance",

        "margin",

        "low_margin_flag",
        "low_confidence_flag",

        "is_anchor_patient",

        "no_face_detected",
    ]

    no_face_count = 0
    low_margin_count = 0
    low_confidence_count = 0

    label_counts = {
        label: 0
        for label in range(1, NUM_CLASSES + 1)
    }

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, record in enumerate(records):

            features = extract_features(
                record["image_path"]
            )

            is_anchor_patient = (
                record["patient_id"]
                in anchor_patient_set
            )

            # ------------------------------------------------
            # No face
            # ------------------------------------------------

            if features is None:

                no_face_count += 1

                writer.writerow({
                    **record,

                    "label": "no_face",

                    "confidence": "",
                    "distance_to_assigned_anchor": "",

                    "second_closest_label": "",
                    "second_closest_distance": "",

                    "margin": "",

                    "low_margin_flag": "",
                    "low_confidence_flag": "",

                    "is_anchor_patient":
                        is_anchor_patient,

                    "no_face_detected": True,
                })

                continue

            # ------------------------------------------------
            # Calculate distances
            # ------------------------------------------------

            distances = calculate_distances(
                features,
                anchor_features,
                feature_min,
                feature_range
            )

            sorted_labels = sorted(
                distances,
                key=distances.get
            )

            best_label = sorted_labels[0]
            second_label = sorted_labels[1]

            best_distance = distances[
                best_label
            ]

            second_distance = distances[
                second_label
            ]

            margin = (
                second_distance -
                best_distance
            )

            confidence = calculate_confidence(
                best_distance,
                second_distance
            )

            low_margin = (
                margin < LOW_MARGIN_THRESHOLD
            )

            low_confidence = (
                confidence < LOW_CONFIDENCE_THRESHOLD
            )

            if low_margin:
                low_margin_count += 1

            if low_confidence:
                low_confidence_count += 1

            label_counts[best_label] += 1

            # ------------------------------------------------
            # Write
            # ------------------------------------------------

            writer.writerow({
                **record,

                "label": best_label,

                "confidence":
                    round(confidence, 5),

                "distance_to_assigned_anchor":
                    round(best_distance, 5),

                "second_closest_label":
                    second_label,

                "second_closest_distance":
                    round(second_distance, 5),

                "margin":
                    round(margin, 5),

                "low_margin_flag":
                    low_margin,

                "low_confidence_flag":
                    low_confidence,

                "is_anchor_patient":
                    is_anchor_patient,

                "no_face_detected":
                    False,
            })

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (index + 1) % 500 == 0:

                print(
                    f"Processed "
                    f"{index + 1:,}/{len(records):,}"
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)

    print(
        f"Total images       : {len(records):,}"
    )

    print(
        f"No-face images     : {no_face_count:,}"
    )

    print(
        f"Low-margin images  : {low_margin_count:,}"
    )

    print(
        f"Low-confidence     : {low_confidence_count:,}"
    )

    print("\nPseudo-label distribution:")

    for label in range(1, NUM_CLASSES + 1):

        print(
            f"Grade {label}: "
            f"{label_counts[label]:,}"
        )

    print(
        f"\nOutput: {OUTPUT_CSV}"
    )

    print("\nIMPORTANT:")
    print(
        "The original train/val/test split was NOT used "
        "to create the final dataset split."
    )

    print(
        "Next step: perform a patient-wise split using "
        "patient_id."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()