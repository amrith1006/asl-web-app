"""
ASL Detection Web App — FastAPI + WebSocket
Imports the detection logic directly from the original app.py
"""

import os
import csv
import base64
import json

import cv2 as cv
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# ─── Import detection functions from the original app.py ────
from app import calc_landmark_list, pre_process_landmark
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# ─── App Setup ──────────────────────────────────────
app = FastAPI(title="ASL Detection")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Load ML Models (once at startup) ───────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

keypoint_classifier = KeyPointClassifier()

# Read labels
with open(
    "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
) as f:
    reader = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in reader]


def get_normalized_landmarks(landmarks):
    """Return landmark positions as normalized [0-1] coordinate pairs for rendering."""
    result = []
    for lm in landmarks.landmark:
        result.append({"x": lm.x, "y": lm.y, "z": lm.z})
    return result


# ─── Routes ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)


# ─── WebSocket Endpoint ─────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            frame_data = payload.get("frame", "")

            if not frame_data:
                continue

            # Decode base64 → image
            try:
                img_bytes = base64.b64decode(frame_data.split(",")[1])
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv.imdecode(nparr, cv.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "Invalid frame"})
                continue

            if image is None:
                continue

            # Process with MediaPipe
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            detections = []

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    normalized_landmarks = get_normalized_landmarks(hand_landmarks)

                    # Use imported functions from app.py
                    landmark_list = calc_landmark_list(image, hand_landmarks)
                    pre_processed = pre_process_landmark(landmark_list)
                    hand_sign_id = keypoint_classifier(pre_processed)

                    hand_label = handedness.classification[0].label
                    confidence = round(handedness.classification[0].score * 100, 1)

                    detections.append({
                        "letter": keypoint_classifier_labels[hand_sign_id],
                        "hand": hand_label,
                        "confidence": confidence,
                        "landmarks": normalized_landmarks,
                    })

            await websocket.send_json({
                "detections": detections,
                "hands_detected": len(detections),
            })

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")


# ─── Run ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
