import urllib.request
import os
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ===================================
# CONFIGURATION
# ===================================
MODEL_PATH = r"C:\Users\bomma\Downloads\ethical-face-anonymization\yolov8s-oiv7.pt"
model = YOLO(MODEL_PATH)

# Online image URLs
ONLINE_IMAGES = [
    "https://images.unsplash.com/photo-1593642532973-d31b6557fa68",
    "https://images.unsplash.com/photo-1517816743773-6e0fd518b4a6",
    "https://images.unsplash.com/photo-1535930749574-1399327ce78f",
    "https://images.unsplash.com/photo-1522202176988-66273c2fd55f",
    "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d"
]

# Local images (add your own)
LOCAL_IMAGES = [
    r"C:\Users\bomma\Downloads\ethical-face-anonymization\online_inference_results\img_local.jpg"
]

# ===================================
# DOWNLOAD ONLINE IMAGES
# ===================================
os.makedirs("temp_images", exist_ok=True)
downloaded_files = []

for i, url in enumerate(ONLINE_IMAGES):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req)
        img_path = f"temp_images/img_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(resp.read())
        downloaded_files.append(img_path)
        print(f"✅ Downloaded: {img_path}")
    except Exception as e:
        print(f"⚠️ Failed to download {url}: {e}")

# ===================================
# COMBINE ONLINE + LOCAL IMAGES
# ===================================
all_images = downloaded_files + LOCAL_IMAGES

# ===================================
# RUN YOLO ON ALL IMAGES
# ===================================
all_predictions = []
os.makedirs("annotated_images", exist_ok=True)

for path in all_images:
    print(f"\n🔍 Running detection on: {path}")
    results = model(path)  # returns a list
    result = results[0]    # ✅ extract the first result

    # Show image (optional)
    result.show()  # comment out if running on a headless system (e.g., remote server)

    # Convert detections to DataFrame
    df = result.pandas().xyxy[0]
    print(df)
    all_predictions.append(df)

    # Save annotated image
    save_path = os.path.join("annotated_images", os.path.basename(path))
    result.save(filename=save_path)
    print(f"🖼️ Saved annotated image: {save_path}")

# ===================================
# SIMPLE METRICS (DEMO)
# ===================================
# ⚠️ Note: Without ground-truth labels, metrics are simulated.

true_labels = [1, 1, 0, 0, 0, 1]  # example ground truth (1 = object detected, 0 = none)
pred_labels = [1 if len(df) > 0 else 0 for df in all_predictions]

# Match the length of lists if mismatch (for safety)
min_len = min(len(true_labels), len(pred_labels))
true_labels = true_labels[:min_len]
pred_labels = pred_labels[:min_len]

precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("\n====== DETECTION METRICS (Simulated Example) ======")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

print("\n✅ All annotated images saved in 'annotated_images' folder.")
