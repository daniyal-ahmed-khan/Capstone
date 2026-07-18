import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import os
import gdown

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Disease Classifier",
    page_icon="🫁",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────
CLASSES = ['No Finding', 'Infiltration', 'Pneumonia']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
MODEL_PATH = 'densenet121_best_checkpoint.pth'
GDRIVE_FILE_ID = '1m0mSgU98VuVEbPO0SiTeROUI2OoEUs0A'

# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights... please wait")
        gdown.download(
            f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}',
            MODEL_PATH, quiet=False
        )

    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 3)
    )
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# ── Preprocessing ─────────────────────────────────────────────────────────
def preprocess_image(image):
    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Week 3 OpenCV pipeline
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    img_denoised = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    img_resized = cv2.resize(img_denoised, (224, 224), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # Normalize manually
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img_normalized = (img_float - mean) / std
    tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0)

    return tensor, img_clahe, img_denoised, img_resized

# ── Main app ──────────────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Disease Classifier")
st.markdown("**CIS-627 Capstone | Daniyal Ahmed Khan | St. Thomas University**")
st.markdown("---")

st.markdown("""
This app classifies chest X-rays into three categories:
- **No Finding** — Normal chest X-ray
- **Infiltration** — Abnormal substance in lung tissue
- **Pneumonia** — Lung infection
""")

st.warning("⚠️ This is a screening tool only. Not a substitute for professional medical diagnosis.")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** DenseNet-121  
    **Pretrained on:** ImageNet  
    **Fine-tuned on:** NIH Chest X-Ray14 + Kaggle Pneumonia  
    **Training images:** 74,503  
    **Overall accuracy:** 84%  
    
    **Per-class F1 scores:**
    - No Finding: 0.90
    - Infiltration: 0.28
    - Pneumonia: 0.96
    """)

    st.header("Model Performance")
    st.progress(0.84, text="Overall Accuracy: 84%")
    st.progress(0.90, text="No Finding F1: 90%")
    st.progress(0.28, text="Infiltration F1: 28%")
    st.progress(0.96, text="Pneumonia F1: 96%")

    st.header("Dataset Info")
    st.markdown("""
    **Primary:** NIH Chest X-Ray14  
    **Supplementary:** Kaggle Pneumonia Dataset  
    **Classes:** No Finding, Infiltration, Pneumonia  
    **Split:** Patient-level 70/15/15
    """)

# ── Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a chest X-ray in PNG or JPEG format"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Preprocess
    with st.spinner("Preprocessing image..."):
        tensor, img_clahe, img_denoised, img_resized = preprocess_image(image)

    # Show preprocessing steps
    with col2:
        st.subheader("After CLAHE Enhancement")
        st.image(img_clahe, use_container_width=True, clamp=True)

    # Show preprocessing pipeline
    st.subheader("OpenCV Preprocessing Pipeline")
    cols = st.columns(4)
    cols[0].image(np.array(image.convert('L')), caption="1. Original", clamp=True)
    cols[1].image(img_clahe, caption="2. After CLAHE", clamp=True)
    cols[2].image(img_denoised, caption="3. After Denoising", clamp=True)
    cols[3].image(img_resized, caption="4. Final (224×224)", clamp=True)

    # Predict
    with st.spinner("Running prediction..."):
        model = load_model()
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            predicted_class = probs.argmax().item()
            confidence = probs[predicted_class].item()

    # Results
    st.markdown("---")
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        if CLASSES[predicted_class] == 'Pneumonia':
            st.error(f"🔴 **{CLASSES[predicted_class]}**")
        elif CLASSES[predicted_class] == 'Infiltration':
            st.warning(f"🟡 **{CLASSES[predicted_class]}**")
        else:
            st.success(f"🟢 **{CLASSES[predicted_class]}**")

        st.metric("Confidence", f"{confidence:.1%}")

    with col2:
        st.subheader("Confidence Scores")
        for i, cls in enumerate(CLASSES):
            prob = probs[i].item()
            st.progress(prob, text=f"{cls}: {prob:.1%}")

    st.markdown("---")
    st.caption("Model: DenseNet-121 | Trained on NIH Chest X-Ray14 + Kaggle Pneumonia Dataset")