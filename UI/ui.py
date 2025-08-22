import streamlit as st
import cv2
import numpy as np
import math
import pickle

# ================================
# Load Logistic Regression Model
# ================================
with open(r"D:\manu project\new\logistic_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    scaler = saved["scaler"]

# ================================
# Preprocessing Functions
# ================================
def remove_background_opencv(img):
    """Remove background using threshold + morphology"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)
    return mask, result

def get_outer_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    contour_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    cv2.drawContours(contour_img, [c], -1, (255, 255, 255), 2)
    return c, contour_img

def estimate_center_and_radii(contour_img, contour, angle_step=10):
    """Estimate center and radii from outer contour image"""
    (cx, cy), _ = cv2.minEnclosingCircle(contour)
    cx, cy = int(cx), int(cy)

    # Binary version of contour-only image
    gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    overlay = contour_img.copy()
    radii = []

    for ang in range(0, 360, angle_step):
        theta = math.radians(ang)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        for r in range(1, int(2 * math.hypot(contour_img.shape[1], contour_img.shape[0]))):
            x = int(round(cx + r * cos_t))
            y = int(round(cy + r * sin_t))
            if x < 0 or y < 0 or x >= contour_img.shape[1] or y >= contour_img.shape[0]:
                break
            if bin_img[y, x] > 0:  # Hit contour
                radii.append(r)
                cv2.line(overlay, (cx, cy), (x, y), (0, 0, 255), 2)
                break

    cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

    if not radii:
        return overlay, 1.0  # avoid division by zero

    r_min, r_max = np.min(radii), np.max(radii)
    r_ratio = r_max / r_min if r_min > 0 else 1.0

    return overlay, r_ratio

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Cup Defect Detection", layout="wide")
st.title("🟢 Cup Defect Detection with Preprocessing Steps")

uploaded_file = st.file_uploader("Upload a cup image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Step 1: Show original
    st.subheader("Step 1: Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)

    # Step 2: Background removal
    st.subheader("Step 2: Background Removed")
    mask, bg_removed = remove_background_opencv(img)
    st.image(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB), width=400)

    # Step 3: Outer contour
    st.subheader("Step 3: Outer Contour")
    contour, contour_img = get_outer_contour(mask)
    if contour is None:
        st.error("No contour detected in this image.")
    else:
        st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), width=400)

        # Step 4: Center & radii overlay
        st.subheader("Step 4: Center and Radii (Contour-based)")
        overlay_img, r_ratio = estimate_center_and_radii(contour_img, contour)
        st.image(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), width=400)

        # Step 5: Prediction
        st.subheader("Step 5: Prediction Result")
        X_scaled = scaler.transform([[r_ratio]])
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]

        label = "Non-defected ✅" if pred == 1 else "Defected 🚨"
        confidence = prob[pred] * 100

        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.markdown(f"**R Ratio (r_max/r_min):** {r_ratio:.4f}")
