
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound
import threading
import time
import os
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ------------------- CONFIG -------------------
camera_url = "http://10.99.43.223:4747/video"
model_path = r"C:\Users\bomma\Downloads\ethical-face-anonymization\yolov8s-oiv7.pt"
LARGE_OBJECT_THRESHOLD = 2.5
SMALL_OBJECT_THRESHOLD = 0.9
SPEECH_INTERVAL = 10.0
TTS_CACHE_DIR = "tts_cache"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# ------------------- LOAD YOLO MODEL -------------------
print("object detection setup....")
model = YOLO(model_path)
print(f" Model loaded successfully with {len(model.names)} classes.")

# ------------------- BAYESIAN NETWORK -------------------
bn_model = BayesianNetwork([
    ('ObjectType', 'Risk'),
    ('Confidence', 'Risk'),
    ('Proximity', 'Risk')
])

cpd_object = TabularCPD('ObjectType', 3, [[0.4], [0.4], [0.2]])
cpd_conf = TabularCPD('Confidence', 3, [[0.3], [0.4], [0.3]])
cpd_prox = TabularCPD('Proximity', 3, [[0.3], [0.4], [0.3]])

risk_vals = []
for obj in range(3):
    for conf in range(3):
        for prox in range(3):
            if prox == 2:
                if obj == 0:
                    risk_vals.append([0.1,0.3,0.6])
                elif obj == 1:
                    risk_vals.append([0.1,0.2,0.7])
                else:
                    risk_vals.append([0.2,0.3,0.5])
            elif prox == 1:
                if obj == 0:
                    risk_vals.append([0.3,0.4,0.3])
                else:
                    risk_vals.append([0.5,0.4,0.1])
            else:
                risk_vals.append([0.7,0.2,0.1])
risk_vals = np.array(risk_vals).T.tolist()

cpd_risk = TabularCPD('Risk', 3, values=risk_vals,
                      evidence=['ObjectType','Confidence','Proximity'],
                      evidence_card=[3,3,3])

bn_model.add_cpds(cpd_object, cpd_conf, cpd_prox, cpd_risk)
if not bn_model.check_model():
    raise RuntimeError("Bayesian Network is invalid!")

inference = VariableElimination(bn_model)

# ------------------- VOICE ALERT SYSTEM -------------------
active_alerts = {}

def speak(message):
    safe_name = "".join(c for c in message if c.isalnum() or c==' ').strip().replace(' ','_')
    filepath = os.path.join(TTS_CACHE_DIR, f"{safe_name}.mp3")
    if not os.path.exists(filepath):
        tts = gTTS(message)
        tts.save(filepath)
    playsound(filepath)

def continuous_alert(label, distance):
    while active_alerts.get(label, False):
        speak(f"{label} is {distance:.1f} meters away")
        time.sleep(SPEECH_INTERVAL)

def start_alert(label, distance):
    if label not in active_alerts or not active_alerts[label]:
        active_alerts[label] = True
        threading.Thread(target=continuous_alert, args=(label, distance), daemon=True).start()

def stop_alert(label):
    active_alerts[label] = False

# ------------------- METRICS -------------------
total_frames = 0
objects_detected = 0
alerts_triggered = 0
false_alerts = 0
fps_list = []

# ------------------- MAIN LOOP -------------------
cap = cv2.VideoCapture(camera_url)
if not cap.isOpened():
    raise RuntimeError(" Cannot open camera stream. Check your IP Webcam URL.")

print("🎥 Object detection started — press 'q' or Ctrl+C to quit.\n")

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]
        total_frames += 1
        objects_detected += len(results.boxes)
        current_near = set()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            conf = float(box.conf[0])

            box_height_ratio = (y2 - y1) / frame.shape[0]
            if box_height_ratio < 0.02:
                continue
            distance = round(0.6 / (box_height_ratio + 1e-3), 2)

            if any(word in label for word in ['car','bus','truck','motorbike']):
                obj_type = 1
            elif any(word in label for word in ['person','man','woman']):
                obj_type = 2
            else:
                obj_type = 0

            conf_bucket = 0 if conf < 0.4 else 1 if conf < 0.7 else 2

            if obj_type == 0:
                if distance <= LARGE_OBJECT_THRESHOLD:
                    prox_bucket = 2
                elif distance <= LARGE_OBJECT_THRESHOLD*1.5:
                    prox_bucket = 1
                else:
                    prox_bucket = 0
            else:
                if distance <= SMALL_OBJECT_THRESHOLD:
                    prox_bucket = 2
                elif distance <= SMALL_OBJECT_THRESHOLD*1.5:
                    prox_bucket = 1
                else:
                    prox_bucket = 0

            q = inference.query(['Risk'], evidence={
                'ObjectType': obj_type,
                'Confidence': conf_bucket,
                'Proximity': prox_bucket
            })
            p_high = float(q.values[2])

            alert_condition = False
            if obj_type == 0 and distance <= LARGE_OBJECT_THRESHOLD:
                alert_condition = True
            elif obj_type != 0 and distance <= SMALL_OBJECT_THRESHOLD:
                alert_condition = True
            elif p_high > 0.6:
                alert_condition = True

            color = (0,0,255) if alert_condition else (0,255,0)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label} D:{distance:.1f}m", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if alert_condition:
                alerts_triggered += 1
                current_near.add(label)
                if (obj_type == 0 and distance > LARGE_OBJECT_THRESHOLD*1.5) or \
                   (obj_type != 0 and distance > SMALL_OBJECT_THRESHOLD*1.5):
                    false_alerts += 1
                start_alert(label, distance)

        for lbl in list(active_alerts.keys()):
            if lbl not in current_near and active_alerts[lbl]:
                stop_alert(lbl)

        fps = 1.0 / (time.time() - start_time)
        fps_list.append(fps)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Blind Assistance + Bayesian Risk", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n Stopped manually with Ctrl+C")

finally:
    # ------------------- METRICS SUMMARY -------------------
    avg_fps = np.mean(fps_list) if fps_list else 0
    precision = ((alerts_triggered - false_alerts) / alerts_triggered * 100) if alerts_triggered else 0

    print("\n PERFORMANCE METRICS ")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Objects Detected: {objects_detected}")
    print(f"Alerts Triggered: {alerts_triggered}")
    print(f"False Alerts: {false_alerts}")
    print(f"Alert Precision: {precision:.2f}%")
    print(f"Average FPS: {avg_fps:.2f}")

    # Cleanup
    for lbl in active_alerts:
        active_alerts[lbl] = False
    cap.release()
    cv2.destroyAllWindows()