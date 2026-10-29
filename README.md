Markdown · README.md

# Facial Paralysis Detection & Recovery Tracking

An Android application that uses an on-device deep learning pipeline to detect facial paralysis, grade its severity on the clinical **House-Brackmann (HB) Scale**, explain its predictions, and track a patient's recovery over time through a personalized **Digital Twin Timeline**.

**Final Year Capstone Project — Batch 201** Department of Computer Science and Engineering, PES University

**Project Guide:** Dr. Pooja Agarwal, Professor, Dept. of CSE

**Team:**

- Abhishek P
- Bhuvi Prashanth
- Harsha
- Shreya Parashar

---

## Overview

Facial paralysis is conventionally diagnosed through subjective visual assessment against the House-Brackmann Scale. This project replaces that subjectivity with an objective, reproducible pipeline: a transfer-learned CNN grades severity directly from a captured facial image, with an auxiliary geometric asymmetry signal guiding the model during training (not fed to it at inference), a region-based explainability layer (LIME) showing *why* a grade was predicted, and a longitudinal Digital Twin view that tracks a patient's grade and asymmetry trend across repeat assessments — all running on-device, with no image leaving the phone.

---

## Core Capabilities

| Capability | Description |
| --- | --- |
| **Severity Grading** | Classifies a captured face into one of 5 House-Brackmann grades (Normal → Severe) |
| **On-device Inference** | Landmark detection and classification run entirely on-device via MediaPipe + TFLite — no server round-trip |
| **Explainable AI (LIME)** | Region-based LIME explanation over landmark-defined face regions (eyes, brows, mouth, cheeks), so predictions are interpretable rather than a black-box label |
| **Digital Twin Timeline** | Built from stored assessment history per patient — visualizes grade and asymmetry trend across visits |
| **Consent-Gated Reporting** | Generates a PDF clinical report (grade, confidence, explanation); explicit photo-inclusion consent is requested on every export, never remembered |
| **Privacy-First Image Storage** | Only two images are retained per patient — baseline (first visit) and latest (most recent) — shown behind a "one view" tap-to-reveal control |
| **Dual Camera Support** | Front and back camera capture with live switching and on-screen lighting/positioning guidance |

---

## System Architecture

A 5-module pipeline, designed for edge deployment and clinical interpretability:

```
1. Data Acquisition
   └─ Front/back camera capture, guided by on-screen lighting/pose instructions

2. Preprocessing
   └─ Face detection, alignment, normalization

3. Feature Extraction & Multi-Task Learning
   ├─ MediaPipe FaceLandmarker → 468-point landmarks → 5 Facial Asymmetry Index (FAI) values
   └─ ResNet50 (ImageNet-pretrained) → Transfer Learning → shared visual embedding
        ├─ Classification Head → House-Brackmann grade (1–5)
        └─ FAI Auxiliary Head (training-only) → predicts FAI from image, discarded at inference

4. AI Models & Digital Twin Timeline
   ├─ Severity Classification (House-Brackmann Scale)
   └─ Assessment History (Room DB) → Digital Twin Timeline View (grade + asymmetry over time)

5. Explainable AI
   ├─ Landmark Importance Mapping
   └─ LIME Explanation (landmark-defined face regions)
```

---

## Model Design Notes

**Transfer Learning.** The backbone is a ResNet50 pretrained on ImageNet, fine-tuned on facial severity data with discriminative learning rates (backbone `1e-5`, new heads `1e-3`) — the pretrained visual features are preserved and gently adapted rather than overwritten, while the newly added classification and auxiliary heads train faster from random initialization.

**FAI as an auxiliary signal, not a classifier input.** An earlier design concatenated the 5 FAI values directly into the classifier. Because those same values were used to generate training labels (see *Labeling Strategy* below), this allowed the model to reconstruct the labeling rule instead of learning from the image — an evaluation leak, confirmed by an implausible \~0.97 validation kappa within 2 epochs. The current design uses FAI only as an auxiliary regression target during training (multi-task learning): the shared visual trunk is also trained to predict the 5 FAI values, which shapes it toward asymmetry-relevant features, but the classification head never receives FAI as input, at training or inference. Only the raw image drives predictions.

**Loss & training setup.** Class-weighted cross-entropy (inverse frequency, correcting for class imbalance) with label smoothing (0.05), combined with a weighted MSE auxiliary loss (`total loss = classification loss + 0.3 × FAI loss`). Model selection and early stopping use **validation Quadratic Weighted Kappa**, not accuracy — kappa is ordinal-aware, penalizing a 5-grade miss far more than a 1-grade miss, matching how clinical severity should actually be scored.

**Patient-wise data splitting.** Train/validation/test splits are performed by patient ID, not by image, with explicit assertions preventing any patient from appearing in more than one split — preventing identity/lighting leakage across splits.

---

## Labeling Strategy

A small set of reference-verified images was used to select 10 **anchor images per grade** (50 total) across the House-Brackmann spectrum. Remaining images were labeled via **nearest-anchor distance** in FAI (geometric asymmetry) feature space — each unlabeled image inherits the grade of its closest anchor. This exemplar-based pseudo-labeling approach was used where exhaustive clinician annotation of the full dataset was not feasible; it is explicitly treated as a weak/reference label rather than verified ground truth, which also motivates evaluating with ordinal-tolerant metrics (±1-grade accuracy, Quadratic Weighted Kappa) alongside exact-match accuracy.

---

## Results

Evaluated on a held-out test set of 13,428 images across 26 patients, never seen during training or validation.

| Metric | Score |
| --- | --- |
| Exact-match accuracy | 70.61% |
| ±1-grade accuracy | 76.56% |
| Macro F1 | 0.6847 |
| Quadratic Weighted Kappa | 0.6485 |

| Grade | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| 1 — Normal | 0.800 | 0.786 | 0.793 | 3309 |
| 2 — Mild | 0.583 | 0.719 | 0.644 | 1948 |
| 3 — Moderate | 0.519 | 0.662 | 0.582 | 1662 |
| 4 — Moderately Severe | 0.549 | 0.681 | 0.608 | 1468 |
| 5 — Severe | 0.937 | 0.713 | 0.810 | 5041 |

A Quadratic Weighted Kappa of 0.65 falls within the "substantial agreement" band (Landis–Koch scale, 0.61–0.80) and is broadly comparable to reported inter-rater agreement between human clinicians scoring the House-Brackmann scale independently — a relevant benchmark given the scale's known subjectivity at adjacent grade boundaries.

**Dataset split:** 35,056 images / 85 patients (fit) · 4,791 images / 15 patients (validation) · 13,428 images / 26 patients (test).

---

## Tech Stack

**Android Application**

- Kotlin, Jetpack Compose (Material3)
- CameraX — front/back capture
- MediaPipe FaceLandmarker — on-device landmark detection
- TensorFlow Lite — on-device severity classifier
- Room — local persistence (patients, assessments, images)
- Native `PdfDocument` + `FileProvider` — report generation and sharing

**Model Training**

- Python, PyTorch, Torchvision (ResNet50)
- MediaPipe FaceMesh — FAI feature extraction
- scikit-learn — evaluation metrics

---

## Limitations & Future Work

- Labels are reference/exemplar-derived, not independently clinician-verified across the full dataset — validating against a clinician-reviewed held-out set is a natural next step.
- Notable confusion between non-adjacent grades (3 and 5) in the confusion matrix suggests static-image geometric asymmetry alone doesn't fully capture HB grade distinctions that depend on *which* movements are preserved (e.g. forehead motion) — incorporating video/dynamic motion cues is a promising extension.
- Classification is currently treated as a standard multi-class problem; an ordinal regression loss (rather than cross-entropy) would more directly optimize for the task's ordinal structure during training, not just at evaluation.

---

## Disclaimer

This application is a research/academic capstone project and a clinical decision-support aid. It is not a diagnostic replacement and should not be used as the sole basis for clinical decisions.
