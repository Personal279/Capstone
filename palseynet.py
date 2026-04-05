"""
PalsyNet Dataset Downloader & Frame Extractor
==============================================
Downloads jasir/palsynet-data (49 videos, video+label columns),
saves raw videos, then extracts frames at 6 FPS into a clean
train/val/test folder structure — no torchcodec needed.

Output:
  palsynet_frames/
  ├── raw_videos/
  │   ├── normal/        *.mp4
  │   └── paralysed/     *.mp4
  ├── frames/
  │   ├── train/
  │   │   ├── normal/    <video_id>/frame_00001.jpg ...
  │   │   └── paralysed/
  │   ├── val/
  │   └── test/
  └── metadata/
      ├── dataset_info.json
      └── extraction_log.csv

Usage:
    python extract_palsynet.py
    python extract_palsynet.py --fps 6 --output ./my_output --split 0.7 0.15 0.15
"""

import os, cv2, csv, json, shutil, argparse, random
from pathlib import Path
from datetime import datetime

try:
    from datasets import load_dataset
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install datasets opencv-python rich huggingface_hub")
    exit(1)

console = Console()

LABEL_MAP = {0: "normal", 1: "paralysed"}   # adjust if dataset uses different ints


# ── Frame extraction ──────────────────────────────────────────────────────────
def extract_frames(video_path: Path, out_dir: Path, target_fps: float, quality: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "frames_saved": 0, "error": "Cannot open video"}

    source_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(source_fps / target_fps))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved = 0
    idx   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            cv2.imwrite(str(out_dir / f"frame_{saved:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        idx += 1
    cap.release()

    return {
        "status": "ok",
        "source_fps": round(source_fps, 2),
        "target_fps": target_fps,
        "frame_interval": frame_interval,
        "total_source_frames": total_frames,
        "frames_saved": saved,
        "resolution": f"{width}x{height}",
    }


# ── Split helper ──────────────────────────────────────────────────────────────
def split_list(items, ratios):
    random.shuffle(items)
    n       = len(items)
    n_train = max(1, round(n * ratios[0]))
    n_val   = round(n * ratios[1]) if n > 1 else 0
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    random.seed(args.seed)
    base_dir     = Path(args.output)
    raw_dir      = base_dir / "raw_videos"
    frames_dir   = base_dir / "frames"
    metadata_dir = base_dir / "metadata"
    for d in [raw_dir, frames_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    split_ratios = tuple(args.split)
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    console.rule("[bold cyan]PalsyNet Dataset Extractor")
    console.print(f"  Output  : [green]{base_dir.resolve()}[/]")
    console.print(f"  FPS     : [yellow]{args.fps}[/]")
    console.print(f"  Quality : [yellow]{args.quality}[/]")
    console.print(f"  Splits  : train={split_ratios[0]} / val={split_ratios[1]} / test={split_ratios[2]}\n")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    console.print("[bold]Step 1/3:[/] Loading dataset…")
    ds = load_dataset("jasir/palsynet-data", verification_mode="no_checks")
    split_data = ds["train"]
    console.print(f"  Rows: [yellow]{len(split_data)}[/]  |  Features: {list(split_data.features.keys())}")

    def resolve_label(val):
        if isinstance(val, str):
            v = val.lower()
            if any(k in v for k in ["palsy", "paralys", "affected", "bells"]):
                return "paralysed"
            return "normal"
        return LABEL_MAP.get(int(val), f"class_{val}")

    # ── 2. Save raw video files via PyArrow (bypasses torchcodec decoder) ─────
    console.print("\n[bold]Step 2/3:[/] Saving raw videos…")

    pa_table = split_data.data.table   # raw PyArrow table — no HF decoding

    video_registry: dict[str, list] = {"normal": [], "paralysed": []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Saving videos…", total=len(split_data))

        for i in range(len(split_data)):
            row_raw   = pa_table.slice(i, 1).to_pydict()
            label_raw = row_raw["label"][0]
            label     = resolve_label(label_raw)
            video_raw = row_raw["video"][0]

            dest_dir = raw_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            saved_path = None

            if isinstance(video_raw, dict):
                src_path  = video_raw.get("path") or ""
                src_bytes = video_raw.get("bytes") or b""

                if src_bytes:
                    ext  = Path(src_path).suffix if src_path else ".mp4"
                    dest = dest_dir / f"video_{i:04d}{ext}"
                    dest.write_bytes(src_bytes)
                    saved_path = dest

                elif src_path and Path(src_path).exists():
                    dest = dest_dir / Path(src_path).name
                    if not dest.exists():
                        shutil.copy2(src_path, dest)
                    saved_path = dest

                else:
                    # repo-relative path — download via hf_hub
                    try:
                        from huggingface_hub import hf_hub_download
                        local = hf_hub_download(
                            repo_id="jasir/palsynet-data",
                            filename=src_path,
                            repo_type="dataset",
                        )
                        dest = dest_dir / Path(src_path).name
                        if not dest.exists():
                            shutil.copy2(local, dest)
                        saved_path = dest
                    except Exception as e:
                        console.print(f"  [yellow]WARN Row {i}: could not download '{src_path}': {e}[/]")

            elif isinstance(video_raw, bytes) and video_raw:
                dest = dest_dir / f"video_{i:04d}.mp4"
                dest.write_bytes(video_raw)
                saved_path = dest

            if saved_path and saved_path.exists():
                video_registry[label].append(saved_path)
                console.print(f"  [dim]{label}/{saved_path.name}[/]")
            else:
                console.print(f"  [red]FAIL Row {i} ({label}): no video data[/]")

            progress.advance(task)

    total = sum(len(v) for v in video_registry.values())
    console.print(f"\n  Saved [green]{total}[/] videos  "
                  f"(normal=[cyan]{len(video_registry['normal'])}[/], "
                  f"paralysed=[cyan]{len(video_registry['paralysed'])}[/])")

    if total == 0:
        console.print("[red]No videos saved — cannot extract frames.[/]")
        return

    # ── 3. Extract frames ─────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 3/3:[/] Extracting frames at [yellow]{args.fps}[/] FPS…")

    log_rows = []
    counts   = {s: {"normal": 0, "paralysed": 0} for s in ["train", "val", "test"]}

    for label, vids in video_registry.items():
        train_v, val_v, test_v = split_list(list(vids), split_ratios)
        for split_name, split_vids in [("train", train_v), ("val", val_v), ("test", test_v)]:
            for vid_path in split_vids:
                vid_id  = vid_path.stem
                out_dir = frames_dir / split_name / label / vid_id
                console.print(f"  [dim]{split_name}/{label}/{vid_id}[/]")
                stats = extract_frames(vid_path, out_dir, args.fps, args.quality)
                counts[split_name][label] += stats.get("frames_saved", 0)
                log_rows.append({
                    "video_id": vid_id, "label": label, "split": split_name,
                    "video_path": str(vid_path), "frames_dir": str(out_dir), **stats
                })

    # ── Metadata ──────────────────────────────────────────────────────────────
    info = {
        "dataset": "jasir/palsynet-data",
        "created_at": datetime.now().isoformat(),
        "target_fps": args.fps,
        "jpeg_quality": args.quality,
        "split_ratios": dict(zip(["train", "val", "test"], split_ratios)),
        "frame_counts": counts,
        "labels": ["normal", "paralysed"],
    }
    (metadata_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    if log_rows:
        with open(metadata_dir / "extraction_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Done!")
    table = Table(title="Frame Extraction Summary", show_lines=True)
    table.add_column("Split",     style="bold cyan")
    table.add_column("Normal",    justify="right")
    table.add_column("Paralysed", justify="right")
    table.add_column("Total",     justify="right", style="bold")
    for s in ["train", "val", "test"]:
        n, p = counts[s]["normal"], counts[s]["paralysed"]
        table.add_row(s, str(n), str(p), str(n + p))
    console.print(table)
    console.print(f"\n  Output: [green]{base_dir.resolve()}[/]")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",     type=float, default=6.0)
    parser.add_argument("--output",  type=str,   default="./palsynet_frames")
    parser.add_argument("--split",   type=float, nargs=3, default=[0.70, 0.15, 0.15])
    parser.add_argument("--quality", type=int,   default=95)
    parser.add_argument("--seed",    type=int,   default=42)
    main(parser.parse_args())