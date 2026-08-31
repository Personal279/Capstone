"""
Facial Palsy Analysis — Pipeline
Data Acquisition -> Preprocessing -> Feature Extraction
PyTorch backend — no TensorFlow dependency.
"""

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime

_DEVICE = torch.device("cpu")


# ── MODULE 1: Data Acquisition ────────────────────────────────────────────────

def create_patient_record(name: str, age: int, side_affected: str, notes: str) -> dict:
    return {
        "patient_id":    f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "name":          name,
        "age":           age,
        "side_affected": side_affected,
        "notes":         notes,
        "timestamp":     datetime.now().isoformat(),
    }


# ── MODULE 2: Preprocessing ───────────────────────────────────────────────────

class FacePreprocessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.left_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml"
        )
        self.right_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_righteye_2splits.xml"
        )
        self.TARGET_SIZE = (224, 224)

    def detect_and_crop_face(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = image_bgr.shape[:2]
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
            )
        if len(faces) == 0:
            return None, None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)
        cropped   = image_bgr[y1:y2, x1:x2]
        bbox_info = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "confidence": 1.0}
        return cropped, bbox_info

    def align_face(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = face_bgr.shape[:2]
        left_eyes  = self.left_eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        right_eyes = self.right_eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
        )
        if len(left_eyes) == 0 or len(right_eyes) == 0:
            return face_bgr, 0.0
        lx, ly, lw, lh = max(left_eyes,  key=lambda e: e[2] * e[3])
        rx, ry, rw, rh = max(right_eyes, key=lambda e: e[2] * e[3])
        left_center  = (lx + lw // 2, ly + lh // 2)
        right_center = (rx + rw // 2, ry + rh // 2)
        if left_center[0] > right_center[0]:
            left_center, right_center = right_center, left_center
        dx    = right_center[0] - left_center[0]
        dy    = right_center[1] - left_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) > 30:
            return face_bgr, 0.0
        eye_midpoint = (
            (left_center[0] + right_center[0]) / 2,
            (left_center[1] + right_center[1]) / 2,
        )
        M       = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
        aligned = cv2.warpAffine(face_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
        return aligned, round(angle, 2)

    def resize_and_normalize(self, face_bgr):
        resized  = cv2.resize(face_bgr, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        face_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        normalized = transform(face_rgb)
        return normalized, face_rgb

    def run(self, image_bgr):
        result = {
            "original": image_bgr, "face_crop": None, "aligned": None,
            "normalized": None, "display_face": None,
            "bbox": None, "rotation_angle": None, "error": None,
        }
        cropped, bbox = self.detect_and_crop_face(image_bgr)
        if cropped is None:
            result["error"] = "No face detected. Use a clear frontal photo with good lighting."
            return result
        result["face_crop"] = cropped
        result["bbox"]      = bbox
        aligned, angle = self.align_face(cropped)
        result["aligned"]        = aligned
        result["rotation_angle"] = angle
        normalized, display_face = self.resize_and_normalize(aligned)
        result["normalized"]   = normalized
        result["display_face"] = display_face
        return result


# ── MODULE 2: Frame Extraction ────────────────────────────────────────────────

class FrameExtractor:
    def __init__(self, every_n_frames=10, max_frames=20):
        self.every_n_frames = every_n_frames
        self.max_frames     = max_frames

    def extract(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {}
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps if fps > 0 else 0
        step            = max(1, total_frames // self.max_frames)
        indices_to_grab = set(range(0, total_frames, step)[:self.max_frames])
        frames        = []
        frame_indices = []
        current       = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current in indices_to_grab:
                frames.append(frame)
                frame_indices.append(current)
            current += 1
        cap.release()
        meta = {
            "total_frames":     total_frames,
            "fps":              round(fps, 2),
            "duration_sec":     round(duration_sec, 2),
            "frames_extracted": len(frames),
            "frame_indices":    frame_indices,
        }
        return frames, meta


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

class FeatureExtractor:
    def __init__(self):
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(base.children())[:-1])
        self.model.to(_DEVICE)
        self.model.eval()

    @torch.no_grad()
    def extract(self, normalized_face) -> np.ndarray:
        if isinstance(normalized_face, np.ndarray):
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(normalized_face)
        else:
            t = normalized_face
        batch  = t.unsqueeze(0).to(_DEVICE)
        output = self.model(batch)
        return output.squeeze().cpu().numpy()

    @torch.no_grad()
    def extract_from_frames(self, normalized_faces: list):
        all_embeddings = [self.extract(f) for f in normalized_faces]
        averaged       = np.mean(all_embeddings, axis=0)
        return averaged, all_embeddings

    def summarize(self, embedding: np.ndarray) -> dict:
        return {
            "dimensions":  len(embedding),
            "mean":        round(float(np.mean(embedding)), 4),
            "std":         round(float(np.std(embedding)),  4),
            "min":         round(float(np.min(embedding)),  4),
            "max":         round(float(np.max(embedding)),  4),
            "nonzero_pct": round(
                float(np.count_nonzero(embedding) / len(embedding) * 100), 1
            ),
        }