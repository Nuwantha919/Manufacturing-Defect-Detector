import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import math

# ------------------- CONFIG -------------------
SUBFOLDERS = {"defected": 0, "non_defected": 1}
ANGLE_STEP = 10
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
USE_HOUGH = False
OUTPUT_BASE_DIR = Path(r"D:\manu project\new")  # Output folder for CSV and annotated images
ANNOTATED_DIR = OUTPUT_BASE_DIR / "annotated"

# ------------------- STEP 1: BACKGROUND REMOVAL -------------------
def remove_background(img_path):
    img = cv2.imread(img_path)
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(output, output, mask=mask)
    return img, mask, result

def save_background_removed_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label_dir in SUBFOLDERS.keys():
        in_path = os.path.join(input_dir, label_dir)
        out_path = os.path.join(output_dir, label_dir)
        os.makedirs(out_path, exist_ok=True)
        for img_name in os.listdir(in_path):
            img_path = os.path.join(in_path, img_name)
            _, _, result = remove_background(img_path)
            cv2.imwrite(os.path.join(out_path, img_name), result)
    print(f"[STEP 1] Background-removed dataset saved at: {output_dir}")

# ------------------- STEP 2: OUTER CONTOUR -------------------
def largest_external_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("No contour found.")
    return max(cnts, key=cv2.contourArea)

def save_outer_contour_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label_dir in SUBFOLDERS.keys():
        in_path = os.path.join(input_dir, label_dir)
        out_path = os.path.join(output_dir, label_dir)
        os.makedirs(out_path, exist_ok=True)
        for fname in os.listdir(in_path):
            img_path = os.path.join(in_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                outer_contour = max(contours, key=cv2.contourArea)
                contour_img = np.zeros_like(img)
                cv2.drawContours(contour_img, [outer_contour], -1, (255,255,255), 2)
                cv2.imwrite(os.path.join(out_path, fname), contour_img)
    print(f"[STEP 2] Outer contour dataset saved at: {output_dir}")

# ------------------- STEP 3 & 4: RADIUS MEASUREMENT -------------------
def estimate_center_and_radius(cnt):
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return (float(cx), float(cy)), float(r)

def ray_hit_point(edge_img, center, theta_deg, max_step=None):
    h, w = edge_img.shape[:2]
    cx, cy = center
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    if max_step is None:
        max_step = int(2 * math.hypot(w, h))
    for r in range(1, max_step):
        x = int(round(cx + r * cos_t))
        y = int(round(cy + r * sin_t))
        if x < 0 or y < 0 or x >= w or y >= h:
            return None, float("nan")
        if edge_img[y, x] > 0:
            return (x, y), float(r)
    return None, float("nan")

def process_image(img_path, angle_step=ANGLE_STEP, use_hough=USE_HOUGH):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_thick = cv2.dilate(bin_img, np.ones((3,3), np.uint8), iterations=1)
    cnt = largest_external_contour(bin_thick)
    (cx, cy), _ = estimate_center_and_radius(cnt)
    center = (cx, cy)
    angles = list(range(0, 360, angle_step))
    radii = []
    overlay = img.copy()
    for ang in angles:
        hit, r = ray_hit_point(bin_thick, center, ang)
        radii.append(r)
        if hit is not None:
            cv2.line(overlay, (int(round(cx)), int(round(cy))), hit, (0,255,0), 1)
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), 3, (0,0,255), -1)
    out_img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return out_img, (cx, cy), angles, radii

# ------------------- STEP 5: PROCESS DATASET & SAVE CSV -------------------
def generate_final_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    OUT_IMG_DIR = output_dir / "annotated"
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    rows = []
    angle_cols = [f"r_{a:03d}" for a in range(0, 360, ANGLE_STEP)]
    for label, label_val in SUBFOLDERS.items():
        folder = input_dir / label
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            try:
                annotated, (cx, cy), angles, radii = process_image(str(p))
            except Exception as e:
                print(f"[ERROR] {p}: {e}")
                continue
            rel = p.relative_to(input_dir)
            out_img_path = OUT_IMG_DIR / rel.with_suffix(".png")
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), annotated)
            row = {
                "file": str(rel).replace("\\","/"),
                "defect_status": label_val,
                "center_x": cx,
                "center_y": cy,
            }
            row.update({col:r for col,r in zip(angle_cols,radii)})
            rows.append(row)
    df = pd.DataFrame(rows, columns=["file","defect_status","center_x","center_y"]+angle_cols)
    # compute min, max, ratio
    df["r_min"] = df[angle_cols].min(axis=1)
    df["r_max"] = df[angle_cols].max(axis=1)
    df["r_ratio"] = df["r_max"] / df["r_min"]
    
   # Only keep image filename and r_ratio
    # df_final = df[["file", "r_ratio", "center_x", "center_y"]].copy() # for testing
    df_final = df[["file", "r_ratio", "defect_status"]].copy() # for training
    final_csv = output_dir / "final_data_filewith_ratios.csv"
    df_final.to_csv(final_csv, index=False)
    print(f"[STEP 5] Final CSV saved at: {output_dir / 'final_data_filewith_ratios.csv'}")

# ------------------- MAIN PIPELINE -------------------
def run_pipeline(raw_dataset_dir):
    raw_dataset_dir = Path(raw_dataset_dir)
    cleaned_dir = Path(OUTPUT_BASE_DIR / "dataset_cleaned")
    outer_dir = Path(OUTPUT_BASE_DIR / "dataset_outer_contour")
    save_background_removed_dataset(raw_dataset_dir, cleaned_dir)
    save_outer_contour_dataset(cleaned_dir, outer_dir)
    generate_final_csv(outer_dir, OUTPUT_BASE_DIR)

if __name__ == "__main__":
    dataset_path = r"D:\manu project\new\dataset"  # <- Raw dataset folder
    run_pipeline(dataset_path)
