"""
Dermo-Scope: Real-Time Skin Disease Analysis AR Web Application
-----------------------------------------------------------------
An interactive Streamlit web app for real-time skin disease detection
with Grad-CAM explainability.

Author: Dermo-Scope Team
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from pathlib import Path

# Configuration
# MODEL_PATH = Path(__file__).parent.parent / "model_training" / "skin_model.h5"
MODEL_PATH = Path(__file__).parent.parent / "model_training" / "skin_model.h5"
IMG_SIZE = 224

# Disease class information
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_LABELS = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

# High-risk diseases
HIGH_RISK_CLASSES = ['mel', 'bcc']
MONITOR_CLASSES = ['akiec']

# Color codes (BGR format for OpenCV)
COLOR_HIGH_RISK = (0, 0, 255)    # Red
COLOR_MONITOR = (0, 165, 255)    # Orange
COLOR_LOW_RISK = (0, 255, 0)     # Green


@st.cache_resource
def load_classification_model():
    """Load the trained model (cached for performance)."""
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.info("Please run the training script first: `python model_training/02_train_model.py`")
        st.stop()
    
    try:
        model = load_model(str(MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


def generate_gradcam(model, image, class_idx, layer_name='Conv_1'):
    """
    Generate Grad-CAM heatmap for explainability.
    
    Args:
        model: Trained Keras model
        image: Preprocessed input image (1, 224, 224, 3)
        class_idx: Predicted class index
        layer_name: Name of the convolutional layer to visualize
    
    Returns:
        heatmap: Grad-CAM heatmap (224, 224)
    """
    try:
        # Create a model that outputs both predictions and feature maps
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_idx]
        
        # Get gradients of the loss w.r.t. the convolutional layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all the filters to get the heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        
        return heatmap
    
    except Exception as e:
        # If Grad-CAM fails, return empty heatmap
        return np.zeros((IMG_SIZE, IMG_SIZE))


def apply_heatmap_overlay(image, heatmap, alpha=0.4):
    """
    Apply heatmap overlay on the image.
    
    Args:
        image: Original image (BGR)
        heatmap: Grad-CAM heatmap (0-1 range)
        alpha: Transparency factor
    
    Returns:
        overlayed_image: Image with heatmap overlay
    """
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Blend with original image
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


class VideoTransformer(VideoProcessorBase):
    """
    Video processor for real-time skin disease detection.
    """
    
    def __init__(self):
        self.model = load_classification_model()
        self.prediction_text = ""
        self.confidence = 0.0
        self.disease_class = ""
    
    def recv(self, frame):
        """
        Process each video frame.
        
        Args:
            frame: Input video frame
        
        Returns:
            Processed frame with predictions and overlays
        """
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Resize for display
        display_img = cv2.resize(img, (640, 480))
        
        # Prepare image for prediction
        input_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img_normalized = input_img_rgb / 255.0
        input_img_batch = np.expand_dims(input_img_normalized, axis=0)
        
        # Make prediction
        predictions = self.model.predict(input_img_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        
        disease_code = CLASS_NAMES[class_idx]
        disease_name = CLASS_LABELS[disease_code]
        
        # Determine risk level
        if disease_code in HIGH_RISK_CLASSES:
            risk_level = "HIGH RISK"
            text_color = COLOR_HIGH_RISK
            
            # Generate Grad-CAM for high-risk cases
            heatmap = generate_gradcam(self.model, input_img_batch, class_idx)
            
            # Apply heatmap overlay
            heatmap_resized = cv2.resize(heatmap, (640, 480))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            display_img = cv2.addWeighted(display_img, 0.6, heatmap_colored, 0.4, 0)
            
        elif disease_code in MONITOR_CLASSES:
            risk_level = "MONITOR"
            text_color = COLOR_MONITOR
        else:
            risk_level = "LOW RISK"
            text_color = COLOR_LOW_RISK
        
        # Add semi-transparent background for text
        overlay = display_img.copy()
        cv2.rectangle(overlay, (10, 10), (630, 120), (0, 0, 0), -1)
        display_img = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)
        
        # Add text to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Disease name
        cv2.putText(
            display_img,
            f"Disease: {disease_name}",
            (20, 40),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Confidence score
        cv2.putText(
            display_img,
            f"Confidence: {confidence:.1f}%",
            (20, 75),
            font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Risk level with color
        cv2.putText(
            display_img,
            f"Risk: {risk_level}",
            (20, 105),
            font,
            0.7,
            text_color,
            2,
            cv2.LINE_AA
        )
        
        # Convert back to av.VideoFrame
        return av.VideoFrame.from_ndarray(display_img, format="bgr24")


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Dermo-Scope: Skin Disease Detection",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .risk-box {
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .risk-high {
            background-color: #FFEBEE;
            border-left: 4px solid #F44336;
        }
        .risk-monitor {
            background-color: #FFF3E0;
            border-left: 4px solid #FF9800;
        }
        .risk-low {
            background-color: #E8F5E9;
            border-left: 4px solid #4CAF50;
        }
        .info-box {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🔬 Dermo-Scope</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Skin Disease Analysis with AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Risk Legend")
        
        st.markdown("""
            <div class="risk-box risk-high">
                <strong>🔴 HIGH RISK</strong><br>
                Melanoma, Basal Cell Carcinoma<br>
                <small>Immediate medical consultation recommended</small>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="risk-box risk-monitor">
                <strong>🟡 MONITOR</strong><br>
                Actinic Keratoses<br>
                <small>Regular monitoring advised</small>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="risk-box risk-low">
                <strong>🟢 LOW RISK</strong><br>
                Benign conditions<br>
                <small>Generally not concerning</small>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.header("ℹ️ Disease Information")
        
        with st.expander("Melanoma (mel)"):
            st.write("**High Risk** - A serious form of skin cancer that develops in melanocytes.")
        
        with st.expander("Basal Cell Carcinoma (bcc)"):
            st.write("**High Risk** - Most common skin cancer, usually appears on sun-exposed areas.")
        
        with st.expander("Actinic Keratoses (akiec)"):
            st.write("**Monitor** - Precancerous patches caused by sun damage.")
        
        with st.expander("Melanocytic Nevi (nv)"):
            st.write("**Low Risk** - Common moles, usually benign.")
        
        with st.expander("Benign Keratosis (bkl)"):
            st.write("**Low Risk** - Non-cancerous skin growths.")
        
        with st.expander("Dermatofibroma (df)"):
            st.write("**Low Risk** - Benign skin nodules.")
        
        with st.expander("Vascular Lesions (vasc)"):
            st.write("**Low Risk** - Blood vessel abnormalities in the skin.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Live Analysis")
        
        st.markdown("""
            <div class="info-box">
                <strong>📌 Instructions:</strong>
                <ol>
                    <li>Click "START" to activate your webcam</li>
                    <li>Position the skin area in front of the camera</li>
                    <li>View real-time predictions and risk assessment</li>
                    <li>Red heatmap overlay shows AI attention areas for high-risk detections</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        # WebRTC streamer
        webrtc_streamer(
            key="skin-disease-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.header("⚙️ System Info")
        
        # Model status
        if MODEL_PATH.exists():
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not found")
            st.info("Run training script: `python model_training/02_train_model.py`")
        
        st.metric("Input Size", f"{IMG_SIZE}x{IMG_SIZE}")
        st.metric("Classes", len(CLASS_NAMES))
        st.metric("Framework", "TensorFlow + MobileNetV2")
        
        st.markdown("---")
        
        st.header("⚠️ Disclaimer")
        st.warning(
            "This is a prototype AI tool for educational purposes. "
            "It should NOT be used as a substitute for professional medical diagnosis. "
            "Always consult a qualified dermatologist for skin concerns."
        )
        
        st.markdown("---")
        
        st.header("💡 About Grad-CAM")
        st.info(
            "For high-risk predictions, the system displays a **Grad-CAM heatmap** "
            "(red overlay) showing which parts of the image the AI focused on when making its decision. "
            "This provides transparency and helps understand the model's reasoning."
        )


if __name__ == "__main__":
    main()
