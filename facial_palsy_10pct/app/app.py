"""
Facial Palsy Analysis — 10% Implementation Demo
Streamlit web application — supports Image and Video input
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import io
import tempfile
import os

from pipeline import FacePreprocessor, FeatureExtractor, FrameExtractor, create_patient_record

st.set_page_config(
    page_title="Facial Palsy Analysis — 10% Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Facial Palsy Analysis System")
st.caption("10% Implementation — Data Acquisition → Preprocessing → Feature Extraction")
st.divider()

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.header("📋 Module 1 — Patient Info")
    patient_name  = st.text_input("Patient name", placeholder="e.g. John Doe")
    patient_age   = st.number_input("Age", min_value=1, max_value=120, value=35)
    side_affected = st.selectbox("Side affected", ["Left", "Right", "Unknown", "Bilateral"])
    notes         = st.text_area("Clinical notes", placeholder="Optional observations...")
    st.divider()
    st.markdown("**Pipeline stages**")
    st.success("✅ Module 1 — Data acquisition")
    st.success("✅ Module 2 — Preprocessing")
    st.success("✅ Module 2 — Frame extraction")
    st.success("✅ Feature extraction (partial)")
    st.warning("⏳ Severity classification — next phase")
    st.warning("⏳ Multimodal fusion — later")
    st.warning("⏳ Explainable AI — later")

# ── Load models ────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return FacePreprocessor(), FeatureExtractor(), FrameExtractor(every_n_frames=10, max_frames=20)

preprocessor, extractor, frame_extractor = load_models()

# ── Helper: show embedding heatmap ─────────────
def show_embedding_heatmap(embedding, title="Feature embedding heatmap (2048 dimensions)"):
    emb_grid = embedding.reshape(32, 64)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("none")
    im = ax.imshow(emb_grid, aspect="auto", cmap="plasma")
    ax.set_title(title, color="white", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor="none", transparent=True, dpi=120)
    buf.seek(0)
    plt.close()
    return buf

# ── Tabs: Image vs Video ───────────────────────
tab_image, tab_video = st.tabs(["🖼 Image Input", "🎥 Video Input"])


# ══════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════
with tab_image:
    col_upload, col_results = st.columns([1, 2], gap="large")

    with col_upload:
        st.subheader("Upload image")
        uploaded = st.file_uploader(
            "Choose a frontal face photo",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="img_uploader",
        )
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                     caption="Uploaded image", use_column_width=True)
            run_btn = st.button("▶ Run pipeline", type="primary", key="img_run")
        else:
            st.info("Upload a frontal face image to begin.")
            run_btn = False

    with col_results:
        if uploaded and run_btn:
            with st.spinner("Running preprocessing…"):
                prep = preprocessor.run(image_bgr)

            if prep["error"]:
                st.error(prep["error"])
                st.stop()

            with st.spinner("Extracting features…"):
                embedding = extractor.extract(prep["normalized"])
                stats     = extractor.summarize(embedding)

            patient = create_patient_record(
                patient_name or "Unknown", int(patient_age), side_affected, notes
            )

            # Preprocessing steps
            st.markdown("#### Module 2 — Preprocessing")
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            fig.patch.set_facecolor("none")
            for ax, (title, img) in zip(axes, [
                ("Original",     cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
                ("Face crop",    cv2.cvtColor(prep["face_crop"], cv2.COLOR_BGR2RGB)),
                ("Aligned",      cv2.cvtColor(prep["aligned"], cv2.COLOR_BGR2RGB)),
                ("224×224 input",prep["display_face"]),
            ]):
                ax.imshow(img); ax.set_title(title, fontsize=9, color="white"); ax.axis("off")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight",
                        facecolor="none", transparent=True, dpi=120)
            buf.seek(0)
            st.image(buf, use_column_width=True)
            plt.close()

            m1, m2, m3 = st.columns(3)
            m1.metric("Detection confidence", f"{prep['bbox']['confidence']*100:.1f}%")
            m2.metric("Rotation corrected",   f"{prep['rotation_angle']}°")
            m3.metric("Output size",           "224 × 224 px")

            st.divider()
            st.markdown("#### Feature extraction (ResNet50)")
            e1, e2, e3 = st.columns(3)
            e1.metric("Embedding dimensions", stats["dimensions"])
            e2.metric("Mean activation",      stats["mean"])
            e3.metric("Active features",      f"{stats['nonzero_pct']}%")
            st.image(show_embedding_heatmap(embedding), use_column_width=True)

            st.divider()
            st.markdown("#### Module 1 — Patient record")
            patient["embedding_stats"] = stats
            st.json(patient)
            st.download_button(
                "⬇ Download patient record + embedding",
                data=json.dumps({
                    "patient": patient,
                    "embedding_preview": embedding[:20].tolist(),
                    "embedding_stats": stats,
                }, indent=2),
                file_name=f"{patient['patient_id']}_record.json",
                mime="application/json",
            )
        elif not uploaded:
            st.markdown("""
            **What this does:**
            1. Detects and crops the face
            2. Aligns face so eyes are horizontal
            3. Resizes to 224×224 and normalizes
            4. Extracts 2048-dim feature embedding via ResNet50
            """)


# ══════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════
with tab_video:
    col_v_upload, col_v_results = st.columns([1, 2], gap="large")

    with col_v_upload:
        st.subheader("Upload video")
        uploaded_video = st.file_uploader(
            "Choose a face video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
            key="vid_uploader",
        )
        if uploaded_video:
            st.video(uploaded_video)
            run_video_btn = st.button("▶ Run pipeline", type="primary", key="vid_run")
        else:
            st.info("Upload a frontal face video to begin.")
            run_video_btn = False

    with col_v_results:
        if uploaded_video and run_video_btn:

            # Save video to temp file so OpenCV can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            # ── Step 1: Frame extraction ───────
            st.markdown("#### Module 2 — Frame extraction")
            with st.spinner("Extracting frames…"):
                frames, meta = frame_extractor.extract(tmp_path)

            if len(frames) == 0:
                st.error("Could not extract frames from video.")
                os.unlink(tmp_path)
                st.stop()

            v1, v2, v3 = st.columns(3)
            v1.metric("Total frames in video", meta["total_frames"])
            v2.metric("Frames extracted",      meta["frames_extracted"])
            v3.metric("Video duration",        f"{meta['duration_sec']}s")

            # Show sample extracted frames (first 5)
            st.markdown("**Sample extracted frames:**")
            sample_frames = frames[:5]
            fig, axes = plt.subplots(1, len(sample_frames), figsize=(14, 3))
            fig.patch.set_facecolor("none")
            if len(sample_frames) == 1:
                axes = [axes]
            for ax, (i, frm) in zip(axes, enumerate(sample_frames)):
                ax.imshow(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Frame {meta['frame_indices'][i]}", fontsize=8, color="white")
                ax.axis("off")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight",
                        facecolor="none", transparent=True, dpi=100)
            buf.seek(0)
            st.image(buf, use_column_width=True)
            plt.close()

            # ── Step 2: Preprocess + embed each frame ──
            st.divider()
            st.markdown("#### Feature extraction — per frame embeddings")

            progress = st.progress(0, text="Processing frames…")
            per_frame_embeddings = []
            valid_frames = 0

            for i, frame in enumerate(frames):
                prep = preprocessor.run(frame)
                if prep["error"] is None:
                    emb = extractor.extract(prep["normalized"])
                    per_frame_embeddings.append(emb)
                    valid_frames += 1
                progress.progress((i + 1) / len(frames),
                                  text=f"Processing frame {i+1}/{len(frames)}…")

            progress.empty()

            if len(per_frame_embeddings) == 0:
                st.error("No faces detected in any frame. Please use a clearer video.")
                os.unlink(tmp_path)
                st.stop()

            # ── Step 3: Show embedding for each frame individually ──
            f1, f2 = st.columns(2)
            f1.metric("Frames with face detected", valid_frames)
            f2.metric("Embedding dimensions per frame", 2048)

            st.markdown("**Embedding heatmap for each frame:**")
            for i, emb in enumerate(per_frame_embeddings):
                stats = extractor.summarize(emb)
                with st.expander(f"Frame {i+1} — mean activation: {stats['mean']}  |  active features: {stats['nonzero_pct']}%"):
                    st.image(show_embedding_heatmap(
                        emb,
                        title=f"Frame {i+1} — embedding (2048 dimensions)"
                    ), use_column_width=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean", stats["mean"])
                    c2.metric("Std",  stats["std"])
                    c3.metric("Active features", f"{stats['nonzero_pct']}%")

            st.divider()

            # Patient record
            patient = create_patient_record(
                patient_name or "Unknown", int(patient_age), side_affected, notes
            )
            patient["video_meta"]         = meta
            patient["frames_with_face"]   = valid_frames
            patient["per_frame_stats"]    = [extractor.summarize(e) for e in per_frame_embeddings]

            st.markdown("#### Module 1 — Patient record")
            st.json(patient)

            st.download_button(
                "⬇ Download patient record + per-frame embeddings",
                data=json.dumps({
                    "patient": patient,
                    "total_frames_processed": valid_frames,
                    "per_frame_embeddings_preview": [e[:20].tolist() for e in per_frame_embeddings],
                    "per_frame_stats": [extractor.summarize(e) for e in per_frame_embeddings],
                    "note": "Next phase: each frame embedding → severity score → average severity"
                }, indent=2),
                file_name=f"{patient['patient_id']}_video_record.json",
                mime="application/json",
            )

            os.unlink(tmp_path)

        elif not uploaded_video:
            st.markdown("""
            **What this does:**
            1. **Frame extraction** — pulls evenly spaced frames from the video
            2. **Per frame** — detects face, aligns, resizes, extracts 2048-dim embedding
            3. **Averages** all frame embeddings into one final embedding
            4. Shows per-frame activation variance chart

            *Next phase: each frame embedding → severity score → average severity across frames*
            """)