import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="CJ Koi Fish Classifier",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS Part
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --navy:   #0A1628;
    --deep:   #0D1F3C;
    --teal:   #1B6CA8;
    --teal2:  --2485C7;
    --gold:   #D4A843;
    --gold2:  #F0C860;
    --cream:  #F5F0E8;
    --muted:  #8FA3BB;
    --card:   rgba(13,31,60,0.85);
    --glass:  rgba(27,108,168,0.12);
    --radius: 16px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--navy) !important;
    font-family: 'Inter', sans-serif;
    color: var(--cream);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 1.5rem 1rem 4rem !important; max-width: 720px; }

/* ── Animated hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0D1F3C 0%, #1B3A5C 40%, #0A2A4A 70%, #0D1F3C 100%);
    background-size: 300% 300%;
    animation: waterShift 8s ease infinite;
    border: 1px solid rgba(212,168,67,0.3);
    border-radius: var(--radius);
    padding: 2rem 1.5rem 1.8rem;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(212,168,67,0.15) 0%, transparent 65%);
    pointer-events: none;
}
@keyframes waterShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--gold), var(--gold2), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.06em;
    margin: 0 0 0.35rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0;
}

/* ── Section cards ── */
.card {
    background: var(--card);
    border: 1px solid rgba(27,108,168,0.25);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal2);
    margin: 0 0 1rem;
}

/* ── Image preview box ── */
.image-box {
    border: 2px dashed rgba(212,168,67,0.35);
    border-radius: 12px;
    background: rgba(10,22,40,0.6);
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 0.85rem;
    text-align: center;
    padding: 1.5rem;
}
.image-box img { border-radius: 10px; max-width: 100%; }

/* ── Streamlit image override ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid rgba(212,168,67,0.2);
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s ease;
    width: 100%;
    border: none;
    cursor: pointer;
}

/* Upload btn */
button[kind="primary"], div[data-testid="stButton"]:nth-child(1) button {
    background: linear-gradient(135deg, var(--teal), var(--teal2)) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(27,108,168,0.4);
}
button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(27,108,168,0.55) !important;
}

/* Predict btn */
.predict-btn button {
    background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
    color: var(--navy) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(212,168,67,0.4);
}
.predict-btn button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(212,168,67,0.55) !important;
}

/* Remove / Reset btn */
.danger-btn button {
    background: rgba(180,60,60,0.15) !important;
    color: #FF8080 !important;
    border: 1px solid rgba(180,60,60,0.35) !important;
}
.danger-btn button:hover {
    background: rgba(180,60,60,0.28) !important;
}
.reset-btn button {
    background: rgba(100,100,120,0.2) !important;
    color: var(--muted) !important;
    border: 1px solid rgba(100,100,120,0.3) !important;
}
.reset-btn button:hover {
    background: rgba(100,100,120,0.35) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(10,22,40,0.5) !important;
    border: 1.5px dashed rgba(27,108,168,0.4) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] section { background: transparent !important; }

/* ── Alert boxes ── */
.msg-warning {
    background: rgba(255,180,0,0.1);
    border-left: 3px solid #FFB400;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #FFCC55;
    font-size: 0.85rem;
    margin: 0.6rem 0;
}
.msg-error {
    background: rgba(255,80,80,0.1);
    border-left: 3px solid #FF5050;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #FF8888;
    font-size: 0.85rem;
    margin: 0.6rem 0;
}
.msg-success {
    background: rgba(50,200,120,0.1);
    border-left: 3px solid #32C878;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #60DDA0;
    font-size: 0.85rem;
    margin: 0.6rem 0;
}
.msg-info {
    background: rgba(27,108,168,0.15);
    border-left: 3px solid var(--teal2);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #88C4E8;
    font-size: 0.85rem;
    margin: 0.6rem 0;
}

/* ── Prediction result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(13,31,60,0.9), rgba(20,50,85,0.9));
    border: 1px solid rgba(212,168,67,0.4);
    border-radius: var(--radius);
    padding: 1.6rem;
    text-align: center;
    margin-top: 0.8rem;
}
.result-species {
    font-family: 'Cinzel', serif;
    font-size: 1.6rem;
    background: linear-gradient(90deg, var(--gold), var(--gold2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.3rem 0;
}
.result-confidence-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
}
.confidence-bar-track {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.4rem 0 0.8rem;
}
.confidence-bar-fill-good   { background: linear-gradient(90deg,#32C878,#5EE8A0); height:100%; border-radius:999px; }
.confidence-bar-fill-moderate { background: linear-gradient(90deg,#FFB400,#FFD555); height:100%; border-radius:999px; }
.confidence-bar-fill-weak   { background: linear-gradient(90deg,#FF5050,#FF8888); height:100%; border-radius:999px; }

.badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.badge-good     { background:rgba(50,200,120,0.2);  color:#60DDA0; border:1px solid rgba(50,200,120,0.4); }
.badge-moderate { background:rgba(255,180,0,0.2);   color:#FFCC55; border:1px solid rgba(255,180,0,0.4); }
.badge-weak     { background:rgba(255,80,80,0.2);   color:#FF8888; border:1px solid rgba(255,80,80,0.4); }

/* ── Species description box ── */
.species-description {
    background: rgba(10,22,40,0.5);
    border: 1px solid rgba(27,108,168,0.25);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
    text-align: left;
    color: rgba(245,240,232,0.85);
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ── Caution box ── */
.caution-box {
    background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(212,168,67,0.04));
    border: 1px solid rgba(212,168,67,0.3);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-top: 1.2rem;
}
.caution-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.caution-text {
    color: rgba(245,240,232,0.7);
    font-size: 0.82rem;
    line-height: 1.6;
    margin: 0;
}

/* ── Divider ── */
.gold-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,168,67,0.35), transparent);
    margin: 0.2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "warning_msg" not in st.session_state:
    st.session_state.warning_msg = None
if "error_msg" not in st.session_state:
    st.session_state.error_msg = None
if "model" not in st.session_state:
    st.session_state.model = None
if "class_names" not in st.session_state:
    st.session_state.class_names = None
if "file_key" not in st.session_state:
    st.session_state.file_key = 0

# Model loader
@st.cache_resource
def load_model():
    """Load the saved Keras model."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("koi_efficientnet_model.keras")
        return model
    except Exception as e:
        return None

@st.cache_data
def get_class_names():
    """Try to infer class names from Dataset folder, else return placeholder."""
    dataset_path = "Dataset"
    if os.path.isdir(dataset_path):
        names = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
        if names:
            return names
    return None

# Koi variety descriptions
KOI_DESCRIPTIONS = {
    "Asagi": "One of the oldest koi varieties, recognized by its blue-gray net-like scales on the back and red or orange coloration on the belly, fins, and cheeks. High-quality Asagi usually have balanced red markings and a neat blue dorsal pattern.",
    "Bekko": "Simple but elegant koi with a solid base color—white, red, or yellow—covered with black spots. Unlike Utsuri, the black markings appear only on the upper body and do not wrap around.",
    "Doitsu": "German-scaled koi, meaning they have either no scales or a row of large scales along the back and sides. Their smooth skin gives them a distinct glossy appearance.",
    "Ghosiki": "Means \"five colors\" and combines Kohaku-like red patterns over a dark blue, black, or gray netted body. They often develop stronger coloration as they mature.",
    "Goromo": "Koi with Kohaku-style red patterns that have blue or black edging over each red scale, giving the red patches a layered or shaded appearance.",
    "Hikarimoyo": "Metallic-patterned koi with two or more colors, excluding Utsuri and Ogon types. Their shiny scales create a reflective, bright appearance.",
    "Hikarimuji": "Metallic koi with a single solid color, such as platinum, gold, or orange. Their value depends heavily on skin quality and shine consistency.",
    "Hikariutsuri": "Metallic koi with bold black-based patterns combined with white, red, or yellow. They resemble Utsuri but have a metallic shine that makes them stand out more.",
    "Kawarimono": "A broad category for koi that do not fit into the major standard varieties. This group is highly diverse and often includes rare or unique koi.",
    "Kin Gin Rin": "Koi with sparkling reflective scales that glitter under light, regardless of their base variety. The shine can appear silver or gold and is distributed across the body.",
    "Kohaku": "The most famous koi variety, featuring a pure white body with red patterns. It is considered the foundation of many other koi varieties.",
    "Sanke": "Koi with a white body with red and black markings, similar to Kohaku but with additional black spots, usually not appearing on the head.",
    "Showa": "Black-based koi with red and white patterns layered over the black. Unlike Sanke, black is the primary base and often appears on the head and wraps around the body.",
    "Shusui": "The scaleless version of Asagi, featuring smooth skin with a row of blue scales along the back and red or orange on the sides.",
    "Tancho": "Koi with a single red circular marking on the head and a white body. They are highly prized because the marking resembles the Japanese flag.",
    "Utsuri": "Black-based koi with white, red, or yellow patterns that wrap around the body. Their markings are usually larger and more dramatic than Bekko.",
    "Yamato Nishiki": "A metallic version of Sanke, combining white, red, and black patterns with a bright metallic sheen, making them especially eye-catching in ponds.",
}

def get_koi_description(species_name: str) -> str:
    """Return a description for a koi variety, matching loosely on class/folder name."""
    if not species_name:
        return ""
    normalized = species_name.strip().lower().replace("_", " ").replace("-", " ")
    for key, desc in KOI_DESCRIPTIONS.items():
        if key.lower() == normalized:
            return desc
    return ""

# Image validation
def is_color_image(img_array: np.ndarray) -> bool:
    """Return True if image is RGB/BGR (3-channel) or HSV-compatible."""
    return img_array.ndim == 3 and img_array.shape[2] in (3, 4)

def validate_and_prepare(pil_image: Image.Image):
    """
    Validate size and color space.
    Returns (processed_np_array, warning_message, error_message)
    """
    img_array = np.array(pil_image)
    warning = None
    error = None

    # Color space check
    if not is_color_image(img_array):
        error = (
            "⚠️ Unsupported image format — this image appears to be grayscale or "
            "single-channel. Please upload an RGB, BGR, or HSV-compatible color image."
        )
        return None, None, error

    # Strip alpha if RGBA
    if img_array.shape[2] == 4:
        pil_image = pil_image.convert("RGB")
        img_array = np.array(pil_image)
        warning = "ℹ️ RGBA image detected — alpha channel removed. Processing as RGB."

    # Size check
    h, w = img_array.shape[:2]
    if h < 224 or w < 224:
        error = (
            f"⚠️ Image too small — uploaded image is {w}×{h} px. "
            "A minimum of 224×224 pixels is required for reliable classification. "
            "Please upload a larger image."
        )
        return None, None, error

    # Keep original size if image is 224×224 or larger.
    return img_array, warning, None

def preprocess_for_model(img_array: np.ndarray) -> np.ndarray:
    """Prepare image for EfficientNet while keeping uploaded image unchanged."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    if img_array.shape[:2] != (224, 224):
        img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
    img = img_array.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Hero
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🐟 CJ Koi Fish Classifier</p>
    <p class="hero-sub">Identify Koi fishes and get best experience</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()
class_names = get_class_names()

if model is None:
    st.markdown("""<div class="msg-error">
    ⚠️ <strong>Model not found.</strong> Place <code>koi_efficientnet_model.keras</code>
    in the same directory as this app, then reload.
    </div>""", unsafe_allow_html=True)

# Upload section
st.markdown('<div class="card"><p class="card-title">📁 Upload Image</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a koi fish image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    key=f"uploader_{st.session_state.file_key}",
    label_visibility="collapsed",
)

# Handle new upload
if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        processed, warn, err = validate_and_prepare(pil_img)

        if err:
            st.session_state.error_msg = err
            st.session_state.warning_msg = None
            st.session_state.uploaded_image = None
            st.session_state.processed_image = None
            st.session_state.prediction_result = None
        else:
            st.session_state.error_msg = None
            st.session_state.warning_msg = warn
            st.session_state.uploaded_image = pil_img
            st.session_state.processed_image = processed
            st.session_state.prediction_result = None
    except Exception as e:
        st.session_state.error_msg = f"⚠️ Could not read the file: {e}"
        st.session_state.uploaded_image = None

# Show messages inside upload card
if st.session_state.error_msg:
    st.markdown(f'<div class="msg-error">{st.session_state.error_msg}</div>', unsafe_allow_html=True)
if st.session_state.warning_msg:
    st.markdown(f'<div class="msg-warning">{st.session_state.warning_msg}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Image preview box
st.markdown('<div class="card"><p class="card-title">🖼️ Image Preview</p>', unsafe_allow_html=True)
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

if st.session_state.uploaded_image is not None:
    col_img, _ = st.columns([2, 1])
    with col_img:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)
else:
    st.markdown("""
    <div class="image-box">
        <span>No image uploaded yet.<br>
        <small style="opacity:0.6">Supported: JPG · PNG · WEBP · BMP</small></span>
    </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Action buttons
st.markdown('<div class="card"><p class="card-title">⚡ Actions</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.uploaded_image = None
        st.session_state.processed_image = None
        st.session_state.prediction_result = None
        st.session_state.warning_msg = None
        st.session_state.error_msg = None
        st.session_state.file_key += 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if predict_clicked:
    if st.session_state.uploaded_image is None:
        st.markdown("""<div class="msg-error">
        ⚠️ <strong>No image uploaded.</strong> Please upload a koi fish image before predicting.
        </div>""", unsafe_allow_html=True)
    elif model is None:
        st.markdown("""<div class="msg-error">
        ⚠️ Model is unavailable. Cannot run prediction.
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Analysing image…"):
            try:
                img_input = preprocess_for_model(st.session_state.processed_image)
                preds = model.predict(img_input, verbose=0)
                predicted_idx = int(np.argmax(preds))
                confidence = float(np.max(preds)) * 100

                names = class_names if class_names else [f"Class {i}" for i in range(preds.shape[-1])]
                predicted_name = names[predicted_idx] if predicted_idx < len(names) else f"Class {predicted_idx}"

                # Koi detection heuristic: low confidence may mean non-koi
                if confidence < 30:
                    st.session_state.prediction_result = {
                        "type": "non_koi",
                        "confidence": confidence,
                    }
                else:
                    st.session_state.prediction_result = {
                        "type": "koi",
                        "species": predicted_name,
                        "confidence": confidence,
                    }
            except Exception as e:
                st.session_state.prediction_result = {"type": "error", "message": str(e)}

# Result display
result = st.session_state.prediction_result

if result is not None:
    st.markdown('<div class="card"><p class="card-title">📊 Prediction Result</p>', unsafe_allow_html=True)
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    if result["type"] == "error":
        st.markdown(f'<div class="msg-error">⚠️ Prediction failed: {result["message"]}</div>', unsafe_allow_html=True)

    elif result["type"] == "non_koi":
        st.markdown(f"""
        <div class="msg-warning">
        ⚠️ <strong>Identification Failed:</strong> The Application could not confidently identify a koi fish variety in this image. Please upload a clear, well-lit photograph of a koi fish.
        </div>""", unsafe_allow_html=True)

    else:
        conf = result["confidence"]
        species = result["species"]

        if conf >= 70:
            badge_class = "badge-good"
            bar_class   = "confidence-bar-fill-good"
            level_label = "Good Confidence"
            level_icon  = "✅"
        elif conf >= 30:
            badge_class = "badge-moderate"
            bar_class   = "confidence-bar-fill-moderate"
            level_label = "Moderate Confidence"
            level_icon  = "🔶"
        else:
            badge_class = "badge-weak"
            bar_class   = "confidence-bar-fill-weak"
            level_label = "Weak Confidence"
            level_icon  = "⚠️"

        bar_width = f"{conf:.1f}%"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-confidence-label">Identified Species</div>
            <div class="result-species">{species}</div>
            <br>
            <div class="result-confidence-label">Prediction Quality — {conf:.2f}%</div>
            <div class="confidence-bar-track">
                <div class="{bar_class}" style="width:{bar_width};"></div>
            </div>
            <span class="badge {badge_class}">{level_icon}&nbsp; {level_label}</span>
        </div>
        """, unsafe_allow_html=True)

        description = get_koi_description(species)
        if description:
            st.markdown(f"""
            <div class="species-description">
                <strong>About {species}:</strong> {description}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Caution notice
st.markdown("""
<div class="caution-box">
    <div class="caution-title">⚠️&nbsp; Application Performance Notice</div>
    <p class="caution-text">
        <strong>Some times this application fails to perform well in some backgrounds.</strong>
    </p>
</div>
""", unsafe_allow_html=True)