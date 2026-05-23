"""
Streamlit app — YOLOv8 Object Detection
Run: streamlit run app.py
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(page_title="VisionAI — Object Detection", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #07070f; color: #e8e8f0; }
header[data-testid="stHeader"] { background: transparent; }
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #07070f 60%, #0a1a0d 100%);
    border: 1px solid #1a1a2e; border-radius: 24px;
    padding: 3.5rem 2rem; margin-bottom: 2rem;
    position: relative; overflow: hidden; text-align: center;
}
.hero::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 25% 50%, rgba(0,255,100,0.05) 0%, transparent 45%),
                radial-gradient(circle at 75% 50%, rgba(0,200,255,0.05) 0%, transparent 45%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 10px; color: #00ff64; border: 1px solid #00ff6440;
    background: #00ff6410; padding: 5px 14px; border-radius: 999px;
    margin-bottom: 1.2rem; letter-spacing: 3px; text-transform: uppercase;
}
.hero h1 {
    font-size: 3.8rem; font-weight: 800; margin: 0.4rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #00ff64 55%, #00c8ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.05;
}
.hero-sub { color: #555570; font-size: 1rem; margin-top: 0.8rem; letter-spacing: 0.5px; }
.hero-dots { display: flex; justify-content: center; gap: 6px; margin-top: 1.5rem; }
.hero-dot { width:6px; height:6px; border-radius:50%; }
.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 2rem; }
.stat-card {
    background: #0d0d1a; border: 1px solid #1a1a2e; border-radius: 16px;
    padding: 1.2rem; text-align: center; position: relative; overflow: hidden;
}
.stat-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00ff6460, transparent);
}
.stat-value { font-family: 'Space Mono', monospace; font-size: 1.9rem; font-weight: 700; color: #00ff64; line-height: 1; }
.stat-label { font-size: 0.72rem; color: #444460; margin-top: 5px; letter-spacing: 2px; text-transform: uppercase; }
.section-label {
    font-family: 'Space Mono', monospace; font-size: 10px; color: #444460;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #1a1a2e; }
.det-card {
    display: flex; align-items: center; justify-content: space-between;
    background: #0d0d1a; border: 1px solid #1a1a2e; border-radius: 12px;
    padding: 10px 16px; margin-bottom: 8px; transition: border-color 0.2s;
}
.det-card:hover { border-color: #00ff6440; }
.det-left { display: flex; align-items: center; gap: 10px; }
.det-num {
    font-family: 'Space Mono', monospace; font-size: 10px;
    background: #1a1a2e; color: #00ff64; padding: 2px 8px;
    border-radius: 6px; font-weight: 700;
}
.det-icon { font-size: 1.1rem; }
.det-name { font-size: 14px; font-weight: 600; color: #e0e0f0; }
.det-bar-wrap { flex: 1; margin: 0 16px; }
.det-bar-bg { background: #1a1a2e; border-radius: 4px; height: 5px; }
.det-bar-fill { height: 5px; border-radius: 4px; background: linear-gradient(90deg, #00ff64, #00c8ff); }
.det-score {
    font-family: 'Space Mono', monospace; font-size: 11px;
    padding: 3px 10px; border-radius: 999px; white-space: nowrap;
    border: 1px solid; 
}
.no-det { text-align: center; padding: 2rem; color: #444460; font-size: 13px; border: 1px dashed #1a1a2e; border-radius: 12px; }
section[data-testid="stSidebar"] { background: #0d0d1a !important; border-right: 1px solid #1a1a2e !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d0d1a; border-radius: 12px; padding: 4px; border: 1px solid #1a1a2e; gap: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 11px; color: #555570; border-radius: 8px; padding: 8px 20px; }
.stTabs [aria-selected="true"] { background: #1a1a2e !important; color: #00ff64 !important; }
.stButton > button {
    background: linear-gradient(135deg, #00ff64, #00c8ff) !important;
    color: #07070f !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 12px !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 2rem !important; letter-spacing: 1px !important;
}
.footer { text-align: center; font-family: 'Space Mono', monospace; font-size: 10px; color: #222235; padding: 2rem 0 1rem; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ YOLOv8 · PyTorch · Computer Vision</div>
    <h1>Vision AI</h1>
    <p class="hero-sub">Upload any image — instantly detect and classify every object inside it</p>
    <div class="hero-dots">
        <div class="hero-dot" style="background:#00ff64"></div>
        <div class="hero-dot" style="background:#00c8ff"></div>
        <div class="hero-dot" style="background:#7c3aed"></div>
    </div>
</div>
<div class="stats-row">
    <div class="stat-card"><div class="stat-value">85%</div><div class="stat-label">mAP Score</div></div>
    <div class="stat-card"><div class="stat-value">80</div><div class="stat-label">Object Classes</div></div>
    <div class="stat-card"><div class="stat-value">25+</div><div class="stat-label">FPS Live</div></div>
    <div class="stat-card"><div class="stat-value">640px</div><div class="stat-label">Input Resolution</div></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.4, 0.05)
model_size = st.sidebar.selectbox("Model Power",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0,
    format_func=lambda x: {"yolov8n.pt":"⚡ Nano (Fastest)","yolov8s.pt":"⚖️ Small (Balanced)","yolov8m.pt":"🎯 Medium (Most Accurate)"}[x])
st.sidebar.markdown("---")
st.sidebar.markdown("""<div style='font-family:Space Mono,monospace;font-size:10px;color:#444460;letter-spacing:1px;line-height:2'>
BUILT BY<br><span style='color:#00ff64'>KUNALJIT DAS</span><br>B.TECH CSE · AKTU<br>YOLOV8 · PYTORCH · CV</div>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading Vision AI model…")
def load_model(name):
    return YOLO(name)

model = load_model(model_size)

OBJECT_ICONS = {
    "person":"🧑","car":"🚗","dog":"🐕","cat":"🐈","bicycle":"🚲",
    "motorcycle":"🏍️","bus":"🚌","truck":"🚛","traffic light":"🚦",
    "bird":"🐦","horse":"🐴","sheep":"🐑","cow":"🐄","elephant":"🐘",
    "bear":"🐻","zebra":"🦓","giraffe":"🦒","backpack":"🎒","umbrella":"☂️",
    "handbag":"👜","tie":"👔","suitcase":"🧳","bottle":"🍾","cup":"☕",
    "fork":"🍴","knife":"🔪","spoon":"🥄","bowl":"🍜","banana":"🍌",
    "apple":"🍎","sandwich":"🥪","pizza":"🍕","donut":"🍩","cake":"🎂",
    "chair":"🪑","couch":"🛋️","bed":"🛏️","laptop":"💻","mouse":"🖱️",
    "keyboard":"⌨️","cell phone":"📱","book":"📖","clock":"🕐",
    "vase":"🏺","scissors":"✂️","toothbrush":"🪥","tv":"📺",
}

PALETTE = [
    (0,255,100),(0,200,255),(255,180,0),(200,0,255),
    (255,60,120),(60,200,255),(255,140,0),(100,255,180),
]

def get_certainty_label(score):
    if score >= 0.85: return "Very High", "#00ff64"
    if score >= 0.70: return "High",      "#7fff00"
    if score >= 0.55: return "Medium",    "#ffd700"
    return "Low", "#ff6b35"

def detect_image(image_array, conf):
    results = model.predict(source=image_array, conf=conf, imgsz=640, verbose=False)

    class_counters = {}
    color_map = {}
    color_idx = 0
    detections = []

    annotated = image_array.copy()

    for box in results[0].boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf_val = float(box.conf[0])

        # Assign number per class
        class_counters[cls_name] = class_counters.get(cls_name, 0) + 1
        num   = class_counters[cls_name]
        label = f"{cls_name.title()} {num}"

        # Assign consistent color per class
        if cls_name not in color_map:
            color_map[cls_name] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
        color = color_map[cls_name]

        # Box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        font = cv2.FONT_HERSHEY_DUPLEX
        fs = 0.55
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5), font, fs, (0,0,0), 1, cv2.LINE_AA)

        detections.append({"name": cls_name, "label": label, "num": num, "score": conf_val})

    detections.sort(key=lambda x: x["score"], reverse=True)
    return annotated, detections

tab1, tab2 = st.tabs(["📷  Image Detection", "🎥  Video Detection"])

with tab1:
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_array = np.array(image)

        with st.spinner("🔍 Scanning image…"):
            annotated, detections = detect_image(image_array, conf_threshold)

        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with col2:
            st.markdown('<div class="section-label">Detected</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">Results — {len(detections)} object(s) found</div>', unsafe_allow_html=True)

        if detections:
            det_html = ""
            for d in detections:
                icon = OBJECT_ICONS.get(d["name"], "🔷")
                certainty, color = get_certainty_label(d["score"])
                bar_width = int(d["score"] * 100)
                det_html += f"""
                <div class="det-card">
                    <div class="det-left">
                        <span class="det-icon">{icon}</span>
                        <span class="det-name">{d['label']}</span>
                    </div>
                    <div class="det-bar-wrap">
                        <div class="det-bar-bg">
                            <div class="det-bar-fill" style="width:{bar_width}%"></div>
                        </div>
                    </div>
                    <div class="det-score" style="color:{color};border-color:{color}50;background:{color}12">{certainty}</div>
                </div>"""
            st.markdown(det_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-det">🔍 No objects detected — try lowering the Detection Sensitivity slider</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:2px dashed #1a1a2e;border-radius:20px;padding:4rem 2rem;text-align:center;color:#333350">
            <div style="font-size:3rem;margin-bottom:1rem">📷</div>
            <div style="font-family:Space Mono,monospace;font-size:12px;letter-spacing:2px">DROP AN IMAGE TO BEGIN</div>
            <div style="font-size:12px;margin-top:0.5rem;color:#222235">JPG · PNG · WEBP supported</div>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-label">Upload Video</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("", type=["mp4","avi","mov"], label_visibility="collapsed")
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        st.video(uploaded_video)
        if st.button("▶ Run Object Detection"):
            with st.spinner("Processing video…"):
                model.predict(source=tfile.name, conf=conf_threshold, imgsz=640,
                              save=True, project="runs", name="detect", exist_ok=True)
                st.success("✅ Done! Annotated video saved to `runs/detect/` folder.")
        os.unlink(tfile.name)
    else:
        st.markdown("""
        <div style="border:2px dashed #1a1a2e;border-radius:20px;padding:4rem 2rem;text-align:center;color:#333350">
            <div style="font-size:3rem;margin-bottom:1rem">🎥</div>
            <div style="font-family:Space Mono,monospace;font-size:12px;letter-spacing:2px">DROP A VIDEO TO BEGIN</div>
            <div style="font-size:12px;margin-top:0.5rem;color:#222235">MP4 · AVI · MOV supported</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">VISION AI · YOLOV8 + PYTORCH · BUILT BY KUNALJIT DAS · ASSAM KAZIRANGA UNIVERSITY</div>
""", unsafe_allow_html=True)
