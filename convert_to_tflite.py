"""
Convert best_model_experiment_A.pt -- the final, locked-in severity classifier
from today's review -- to TFLite for on-device Android inference.

Pipeline: PyTorch -> ONNX -> TensorFlow SavedModel (via onnx2tf) -> TFLite
(float32 and float16). This mirrors the same conversion path already used to
produce the app's existing resnet50_feat_float16.tflite, so it's a known
route on this toolchain.

IMPORTANT -- read before wiring this into the app:
    The app's current native pipeline is:
        ResNetFeatureExtractor (image -> 2048-dim embedding)
            -> FaiCalculator (image -> 5-dim FAI)
            -> fused -> MlpClassifier (sklearn MLP, mlp.json/scaler.json)
    That is the FAI-fusion architecture from the original proposal. It was
    tried today and rejected: the FAI values are exactly what the labeling
    pipeline used to assign each pseudo-label, so a classifier that sees them
    directly can reconstruct the labeling rule instead of learning from the
    image (see the training postmortem doc for the full writeup).

    This script exports best_model_experiment_A.pt instead: a single,
    self-contained model that goes straight from a preprocessed image to 5
    calibrated class probabilities (softmax baked in). No FAI, no fusion, no
    separate MLP step.

    Wiring this in means REPLACING the FaiCalculator + MlpClassifier +
    scaler.json step, not adding to it. The input convention (224x224, NHWC,
    ImageNet mean/std normalization) matches ResNetFeatureExtractor.kt's
    existing bitmapToInputBuffer() exactly, so that preprocessing code can be
    reused as-is -- only the output parsing changes (5 floats that sum to 1,
    not a 2048-dim embedding needing a separate classifier).

Environment note: this needs its own venv (venv_convert/), separate from the
project's main venv/. Installing tensorflow there pulls in numpy>=2 and
protobuf>=6, which breaks mediapipe's numpy<2/protobuf<5 requirement --
don't install this toolchain into the main venv.

Usage:
    source venv_convert/bin/activate
    python convert_to_tflite.py --checkpoint best_model_experiment_A.pt

Outputs (in tflite_export/):
    severity_model_float32.tflite
    severity_model_float16.tflite
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 5
IMAGE_SIZE = 224
GRADE_NAMES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]


class ResNetOnly(nn.Module):
    """Exact architecture of best_model_experiment_A.pt."""

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


class ExportWrapper(nn.Module):
    """Adds softmax so the exported model outputs calibrated probabilities
    directly -- no softmax needed on the Kotlin side."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        logits = self.model(image)
        return torch.softmax(logits, dim=1)


def export_onnx(checkpoint_path, onnx_path):
    model = ResNetOnly()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    wrapped = ExportWrapper(model)

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        wrapped,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["grade_probabilities"],
        opset_version=13,
        dynamo=False,
    )
    print(f"Exported ONNX: {onnx_path}")
    return wrapped


def convert_onnx_to_saved_model(onnx_path, saved_model_dir):
    if Path(saved_model_dir).exists():
        shutil.rmtree(saved_model_dir)
    subprocess.run(
        [
            "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(saved_model_dir),
            "-osd",
        ],
        check=True,
    )
    print(f"Converted to TF SavedModel: {saved_model_dir}")


def saved_model_to_tflite(saved_model_dir, out_path, float16=False):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    Path(out_path).write_bytes(tflite_model)
    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


def verify(checkpoint_path, tflite_path, label):
    """Runs the same random input through PyTorch and a TFLite model and
    reports the max absolute difference -- catches conversion bugs before
    they turn into a confusing bug hunt inside the Android app.

    IMPORTANT: reloads a fresh model from the checkpoint file rather than
    reusing the object that was passed through torch.onnx.export(). That
    call was found to mutate the model in place (almost certainly perturbing
    BatchNorm running statistics during tracing via the legacy TorchScript
    exporter) -- reusing the post-export object silently compares TFLite
    against an already-corrupted reference, masking real conversion bugs.
    """
    import tensorflow as tf

    model = ResNetOnly()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    wrapped_pytorch_model = ExportWrapper(model)

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (test_input - mean) / std

    with torch.no_grad():
        torch_out = wrapped_pytorch_model(normalized).numpy()[0]

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    nhwc_input = normalized.permute(0, 2, 3, 1).numpy().astype(np.float32)
    interpreter.set_tensor(input_detail["index"], nhwc_input)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_detail["index"]).reshape(-1)

    max_diff = np.max(np.abs(torch_out - tflite_out))
    print(f"\n[{label}] max abs diff vs PyTorch: {max_diff:.6f}")
    print(f"[{label}] PyTorch probs:  {np.round(torch_out, 4)}")
    print(f"[{label}] TFLite  probs:  {np.round(tflite_out, 4)}")
    status = "OK" if max_diff < 0.01 else "CHECK THIS -- diff is larger than expected"
    print(f"[{label}] {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model_experiment_A.pt")
    parser.add_argument("--out_dir", default="tflite_export")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    onnx_path = out_dir / "severity_model.onnx"
    saved_model_dir = out_dir / "saved_model"

    print("=" * 60)
    print("Step 1/4: PyTorch -> ONNX")
    print("=" * 60)
    export_onnx(args.checkpoint, onnx_path)

    print("\n" + "=" * 60)
    print("Step 2/4: ONNX -> TensorFlow SavedModel (onnx2tf)")
    print("=" * 60)
    convert_onnx_to_saved_model(onnx_path, saved_model_dir)

    print("\n" + "=" * 60)
    print("Step 3/4: SavedModel -> TFLite (float32 + float16)")
    print("=" * 60)
    fp32_path = out_dir / "severity_model_float32.tflite"
    fp16_path = out_dir / "severity_model_float16.tflite"
    saved_model_to_tflite(saved_model_dir, fp32_path, float16=False)
    saved_model_to_tflite(saved_model_dir, fp16_path, float16=True)

    print("\n" + "=" * 60)
    print("Step 4/4: Verification against the original PyTorch model")
    print("=" * 60)
    verify(args.checkpoint, fp32_path, "float32")
    verify(args.checkpoint, fp16_path, "float16")

    print("\nDone. Output classes, in order:", GRADE_NAMES)
    print(f"Copy {fp32_path} and/or {fp16_path} into the Android app's assets/ folder.")


if __name__ == "__main__":
    main()
