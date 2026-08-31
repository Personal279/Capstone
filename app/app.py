"""
Facial Palsy Analysis System
Streamlit app — dark clinical UI, no emojis, PyTorch backend
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import io
import tempfile
import os

from pipeline import FacePreprocessor, FeatureExtractor, FrameExtractor, create_patient_record

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PalsyNet — Diagnostic System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #080b10 !important;
    color: #c8d4e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stSidebar"] {
    background: #060810 !important;
    border-right: 1px solid #1a2535 !important;
}

/* ── Scanline overlay ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,200,0.015) 2px,
        rgba(0,255,200,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Header ── */
.sys-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    border-bottom: 1px solid #1a2535;
    padding-bottom: 20px;
    margin-bottom: 8px;
}
.sys-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.02em;
    color: #e8f4ff;
    margin: 0;
}
.sys-header h1 span { color: #00e5b0; }
.sys-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: #00e5b0;
    background: rgba(0,229,176,0.08);
    border: 1px solid rgba(0,229,176,0.25);
    padding: 3px 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.sys-caption {
    font-size: 0.72rem;
    color: #4a6070;
    letter-spacing: 0.06em;
    margin-top: 6px;
    margin-bottom: 28px;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: #0c1118;
    border: 1px solid #1a2535;
    border-top: 2px solid #00e5b0;
    padding: 16px 18px;
    position: relative;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 6px; height: 6px;
    background: #00e5b0;
}
.metric-label {
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    color: #4a6070;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8f4ff;
    line-height: 1;
}
.metric-unit {
    font-size: 0.65rem;
    color: #4a6070;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 14px 0;
}
.section-head-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1a2535, transparent);
}
.section-head-text {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #00e5b0;
    text-transform: uppercase;
    white-space: nowrap;
}
.section-num {
    font-size: 0.6rem;
    color: #1a2535;
    font-weight: 600;
}

/* ── Pipeline status ── */
.pipeline-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #0f1620;
    font-size: 0.72rem;
}
.pipeline-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.pipeline-dot.done  { background: #00e5b0; box-shadow: 0 0 6px #00e5b0; }
.pipeline-dot.next  { background: #f5a623; }
.pipeline-dot.later { background: #1a2535; }
.pipeline-label { color: #8099b0; }
.pipeline-label.done { color: #c8d4e0; }

/* ── Info/warning boxes ── */
.info-box {
    background: rgba(0,229,176,0.04);
    border-left: 3px solid #00e5b0;
    padding: 12px 16px;
    font-size: 0.75rem;
    color: #8099b0;
    margin: 12px 0;
}
.warn-box {
    background: rgba(245,166,35,0.04);
    border-left: 3px solid #f5a623;
    padding: 12px 16px;
    font-size: 0.75rem;
    color: #8099b0;
    margin: 12px 0;
}
.error-box {
    background: rgba(255,70,70,0.06);
    border-left: 3px solid #ff4646;
    padding: 12px 16px;
    font-size: 0.75rem;
    color: #ff9090;
    margin: 12px 0;
}

/* ── Streamlit widget overrides ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div {
    background: #0c1118 !important;
    border: 1px solid #1a2535 !important;
    color: #c8d4e0 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #00e5b0 !important;
    box-shadow: 0 0 0 1px #00e5b020 !important;
}

/* Labels */
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #4a6070 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid #00e5b0 !important;
    color: #00e5b0 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 8px 24px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(0,229,176,0.08) !important;
    box-shadow: 0 0 16px rgba(0,229,176,0.15) !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: #00e5b0 !important;
    color: #080b10 !important;
    font-weight: 600 !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #00ffca !important;
    box-shadow: 0 0 24px rgba(0,229,176,0.35) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0c1118 !important;
    border: 1px dashed #1a2535 !important;
    border-radius: 0 !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00e5b0 !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1a2535 !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #4a6070 !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00e5b0 !important;
    border-bottom-color: #00e5b0 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #0c1118 !important;
    border: 1px solid #1a2535 !important;
    border-top: 2px solid #00e5b020 !important;
    padding: 14px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.6rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #4a6070 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #e8f4ff !important;
}

/* Divider */
hr { border-color: #1a2535 !important; }

/* JSON */
[data-testid="stJson"] {
    background: #0c1118 !important;
    border: 1px solid #1a2535 !important;
    font-size: 0.72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #00e5b0 !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: #00e5b0 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0c1118 !important;
    border: 1px solid #1a2535 !important;
    border-radius: 0 !important;
}

/* Sidebar label */
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.7rem;
    color: #4a6070;
}

/* Images */
[data-testid="stImage"] img {
    border: 1px solid #1a2535;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid #1a2535 !important;
    color: #8099b0 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #00e5b0 !important;
    color: #00e5b0 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stHeader"] { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080b10; }
::-webkit-scrollbar-thumb { background: #1a2535; }
::-webkit-scrollbar-thumb:hover { background: #00e5b0; }
</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0c1118",
    "axes.facecolor":    "#0c1118",
    "axes.edgecolor":    "#1a2535",
    "axes.labelcolor":   "#4a6070",
    "xtick.color":       "#4a6070",
    "ytick.color":       "#4a6070",
    "text.color":        "#c8d4e0",
    "grid.color":        "#1a2535",
    "grid.linestyle":    "--",
})


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sys-header">
    <h1>Palsy<span>Net</span></h1>
    <span class="sys-badge">v0.1 — 10pct impl</span>
</div>
<div class="sys-caption">
    DIAGNOSTIC PIPELINE &nbsp;/&nbsp;
    DATA ACQUISITION &rarr; PREPROCESSING &rarr; FEATURE EXTRACTION
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                color:#e8f4ff;letter-spacing:-0.01em;margin-bottom:20px;">
        Patient Record
    </div>
    """, unsafe_allow_html=True)

    patient_name  = st.text_input("Patient name", placeholder="Full name")
    patient_age   = st.number_input("Age", min_value=1, max_value=120, value=35)
    side_affected = st.selectbox("Affected side", ["Left", "Right", "Unknown", "Bilateral"])
    notes         = st.text_area("Clinical notes", placeholder="Observations...")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.6rem;letter-spacing:0.14em;color:#4a6070;
                text-transform:uppercase;margin-bottom:12px;">Pipeline Status</div>
    <div class="pipeline-item">
        <div class="pipeline-dot done"></div>
        <span class="pipeline-label done">Module 1 — Data acquisition</span>
    </div>
    <div class="pipeline-item">
        <div class="pipeline-dot done"></div>
        <span class="pipeline-label done">Module 2 — Preprocessing</span>
    </div>
    <div class="pipeline-item">
        <div class="pipeline-dot done"></div>
        <span class="pipeline-label done">Module 2 — Frame extraction</span>
    </div>
    <div class="pipeline-item">
        <div class="pipeline-dot done"></div>
        <span class="pipeline-label done">Feature extraction (partial)</span>
    </div>
    <div class="pipeline-item">
        <div class="pipeline-dot next"></div>
        <span class="pipeline-label">Severity classification</span>
    </div>
    <div class="pipeline-item">
        <div class="pipeline-dot later"></div>
        <span class="pipeline-label">Multimodal fusion</span>
    </div>
    <div class="pipeline-item" style="border:none">
        <div class="pipeline-dot later"></div>
        <span class="pipeline-label">Explainable AI</span>
    </div>
    """, unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing models...")
def load_models():
    return (
        FacePreprocessor(),
        FeatureExtractor(),
        FrameExtractor(every_n_frames=10, max_frames=20),
    )

preprocessor, extractor, frame_extractor = load_models()


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_heatmap(embedding, title="Feature embedding — 2048 dimensions"):
    emb_grid = embedding.reshape(32, 64)
    fig, ax  = plt.subplots(figsize=(10, 2.2))
    im = ax.imshow(emb_grid, aspect="auto", cmap="plasma", interpolation="nearest")
    ax.set_title(title, fontsize=8, color="#4a6070", pad=8, loc="left",
                 fontfamily="monospace")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.008)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    plt.close()
    return buf


def section(label, num=""):
    st.markdown(f"""
    <div class="section-head">
        <span class="section-num">{num}</span>
        <span class="section-head-text">{label}</span>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)


def info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def error_box(text):
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_image, tab_video = st.tabs(["Image Input", "Video Input"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    col_upload, col_results = st.columns([1, 2], gap="large")

    with col_upload:
        section("Input", "01")
        uploaded = st.file_uploader(
            "Frontal face photograph",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
            key="img_uploader",
        )
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                     caption="Input image", use_column_width=True)
            run_btn = st.button("Run pipeline", type="primary", key="img_run")
        else:
            info("Upload a clear frontal face photograph to begin analysis.")
            run_btn = False

    with col_results:
        if uploaded and run_btn:

            section("Preprocessing", "02")
            with st.spinner("Detecting and aligning face..."):
                prep = preprocessor.run(image_bgr)

            if prep["error"]:
                error_box(prep["error"])
                st.stop()

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            steps = [
                ("Original",      cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
                ("Face crop",     cv2.cvtColor(prep["face_crop"], cv2.COLOR_BGR2RGB)),
                ("Aligned",       cv2.cvtColor(prep["aligned"], cv2.COLOR_BGR2RGB)),
                ("224x224 input", prep["display_face"]),
            ]
            for ax, (title, img) in zip(axes, steps):
                ax.imshow(img)
                ax.set_title(title, fontsize=7, color="#4a6070", pad=6, fontfamily="monospace")
                ax.axis("off")
            plt.tight_layout(pad=0.4)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
            buf.seek(0)
            st.image(buf, use_column_width=True)
            plt.close()

            c1, c2, c3 = st.columns(3)
            c1.metric("Detection confidence", f"{prep['bbox']['confidence']*100:.0f}%")
            c2.metric("Rotation corrected",   f"{prep['rotation_angle']}deg")
            c3.metric("Output resolution",    "224 x 224")

            section("Feature Extraction — ResNet50", "03")
            with st.spinner("Extracting 2048-dim embedding..."):
                embedding = extractor.extract(prep["normalized"])
                stats     = extractor.summarize(embedding)

            e1, e2, e3 = st.columns(3)
            e1.metric("Embedding dims",  stats["dimensions"])
            e2.metric("Mean activation", stats["mean"])
            e3.metric("Active features", f"{stats['nonzero_pct']}%")
            st.image(make_heatmap(embedding), use_column_width=True)

            section("Patient Record", "04")
            patient = create_patient_record(
                patient_name or "Unknown", int(patient_age), side_affected, notes
            )
            patient["embedding_stats"] = stats
            st.json(patient)

            st.download_button(
                "Download record + embedding",
                data=json.dumps({
                    "patient": patient,
                    "embedding_preview": embedding[:20].tolist(),
                    "embedding_stats": stats,
                }, indent=2),
                file_name=f"{patient['patient_id']}_record.json",
                mime="application/json",
            )

        elif not uploaded:
            section("Pipeline overview", "")
            st.markdown("""
            <div style="font-size:0.75rem;color:#4a6070;line-height:2;">
            01 &nbsp; Face detection and bounding box extraction<br>
            02 &nbsp; Eye-alignment correction via affine transform<br>
            03 &nbsp; Resize to 224x224, ImageNet normalization<br>
            04 &nbsp; ResNet50 backbone &rarr; 2048-dim feature embedding
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    col_v_upload, col_v_results = st.columns([1, 2], gap="large")

    with col_v_upload:
        section("Input", "01")
        uploaded_video = st.file_uploader(
            "Face video file",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="visible",
            key="vid_uploader",
        )
        if uploaded_video:
            st.video(uploaded_video)
            run_video_btn = st.button("Run pipeline", type="primary", key="vid_run")
        else:
            info("Upload a frontal face video to begin frame-level analysis.")
            run_video_btn = False

    with col_v_results:
        if uploaded_video and run_video_btn:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            section("Frame Extraction", "02")
            with st.spinner("Extracting frames..."):
                frames, meta = frame_extractor.extract(tmp_path)

            if len(frames) == 0:
                error_box("Could not extract frames. Please check the video file.")
                os.unlink(tmp_path)
                st.stop()

            v1, v2, v3 = st.columns(3)
            v1.metric("Total frames",     meta["total_frames"])
            v2.metric("Frames extracted", meta["frames_extracted"])
            v3.metric("Duration",         f"{meta['duration_sec']}s")

            sample = frames[:5]
            fig, axes = plt.subplots(1, len(sample), figsize=(14, 3))
            if len(sample) == 1:
                axes = [axes]
            for ax, (i, frm) in zip(axes, enumerate(sample)):
                ax.imshow(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                ax.set_title(f"f{meta['frame_indices'][i]}", fontsize=7,
                             color="#4a6070", fontfamily="monospace")
                ax.axis("off")
            plt.tight_layout(pad=0.4)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            buf.seek(0)
            st.image(buf, use_column_width=True)
            plt.close()

            section("Per-frame Embeddings", "03")
            progress = st.progress(0, text="Processing frames...")
            per_frame_embeddings = []
            valid_frames = 0

            for i, frame in enumerate(frames):
                prep = preprocessor.run(frame)
                if prep["error"] is None:
                    emb = extractor.extract(prep["normalized"])
                    per_frame_embeddings.append(emb)
                    valid_frames += 1
                progress.progress((i + 1) / len(frames),
                                  text=f"Frame {i+1} / {len(frames)}")
            progress.empty()

            if len(per_frame_embeddings) == 0:
                error_box("No faces detected in any frame. Use a clearer video.")
                os.unlink(tmp_path)
                st.stop()

            f1, f2 = st.columns(2)
            f1.metric("Frames with face", valid_frames)
            f2.metric("Dims per frame",   2048)

            for i, emb in enumerate(per_frame_embeddings):
                s = extractor.summarize(emb)
                with st.expander(
                    f"Frame {i+1}  |  mean {s['mean']}  |  active {s['nonzero_pct']}%"
                ):
                    st.image(make_heatmap(emb, f"Frame {i+1} — embedding"),
                             use_column_width=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean", s["mean"])
                    c2.metric("Std",  s["std"])
                    c3.metric("Active", f"{s['nonzero_pct']}%")

            section("Patient Record", "04")
            patient = create_patient_record(
                patient_name or "Unknown", int(patient_age), side_affected, notes
            )
            patient["video_meta"]       = meta
            patient["frames_with_face"] = valid_frames
            patient["per_frame_stats"]  = [extractor.summarize(e)
                                           for e in per_frame_embeddings]
            st.json(patient)

            st.download_button(
                "Download record + per-frame embeddings",
                data=json.dumps({
                    "patient": patient,
                    "total_frames_processed": valid_frames,
                    "per_frame_embeddings_preview": [e[:20].tolist()
                                                     for e in per_frame_embeddings],
                    "per_frame_stats": patient["per_frame_stats"],
                    "note": "Next: frame embedding -> severity score -> averaged severity",
                }, indent=2),
                file_name=f"{patient['patient_id']}_video_record.json",
                mime="application/json",
            )
            os.unlink(tmp_path)

        elif not uploaded_video:
            section("Pipeline overview", "")
            st.markdown("""
            <div style="font-size:0.75rem;color:#4a6070;line-height:2;">
            01 &nbsp; Evenly-spaced frame extraction from video<br>
            02 &nbsp; Per-frame face detection, alignment, normalization<br>
            03 &nbsp; ResNet50 &rarr; 2048-dim embedding per frame<br>
            04 &nbsp; Next phase: per-frame severity &rarr; averaged severity score
            </div>
            """, unsafe_allow_html=True)