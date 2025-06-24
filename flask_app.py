# flask_app.py
from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Constants
BRIGHT_LOW = 135
BRIGHT_HIGH = 190
SHADOW_THRESHOLD = 15
FACE_AREA_THRESHOLD = 0.1
TOLERANCE = 5
YAW_LIMIT = 10
PITCH_LIMIT = 10
ROLL_LIMIT = 10

# 3D model points
model_points = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype="double")

LANDMARK_IDXS = {
    "nose": 1, "chin": 152,
    "left_eye": 263, "right_eye": 33,
    "left_mouth": 287, "right_mouth": 57
}

def estimate_head_pose(landmarks, shape):
    h, w = shape[:2]
    image_points = np.array([
        [landmarks[LANDMARK_IDXS[key]].x * w, landmarks[LANDMARK_IDXS[key]].y * h]
        for key in LANDMARK_IDXS
    ], dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    _, rvec, tvec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rvec)
    pose = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose)
    yaw, pitch, roll = euler[1][0], euler[0][0], euler[2][0]
    return yaw, pitch, roll

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        message, guidance = "", ""
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            points = np.array([(int(pt.x * w), int(pt.y * h)) for pt in lm])
            hull = cv2.convexHull(points)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)

            area_ratio = np.sum(mask > 0) / (h * w)
            if area_ratio < FACE_AREA_THRESHOLD:
                message = "Move closer"
            else:
                yaw, pitch, roll = estimate_head_pose(lm, frame.shape)
                if abs(yaw) > YAW_LIMIT + TOLERANCE:
                    guidance += "Turn left | " if yaw < 0 else "Turn right | "
                if abs(pitch) > PITCH_LIMIT + TOLERANCE:
                    guidance += "Tilt down | " if pitch < 0 else "Tilt up | "
                if 180 - abs(roll) > ROLL_LIMIT + TOLERANCE and abs(roll) > 90:
                    guidance += "Tilt right | " if roll < 0 else "Tilt left | "
                if abs(roll) > ROLL_LIMIT + TOLERANCE and abs(roll) < 90:
                    guidance += "Tilt right | " if roll < 0 else "Tilt left | "
                if guidance == "":
                    guidance = "Face aligned | "

                v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
                avg_brightness = np.mean(v[mask > 0])
                if avg_brightness < BRIGHT_LOW:
                    message = "Increase lighting"
                elif avg_brightness > BRIGHT_HIGH:
                    message = "Too much lighting"
                else:
                    message = "Lighting OK"

                mid_x = np.mean(points[:, 0])
                left_mask = (mask > 0) & (np.arange(w)[None, :] < mid_x)
                right_mask = (mask > 0) & (np.arange(w)[None, :] >= mid_x)
                if left_mask.any() and right_mask.any():
                    left_b = np.mean(v[left_mask])
                    right_b = np.mean(v[right_mask])
                    if abs(left_b - right_b) > SHADOW_THRESHOLD:
                        side = "left" if left_b < right_b else "right"
                        message += f" | Shadow on {side}"
                        guidance += f"Light your {side} side | "

                cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, guidance.strip(" | "), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return '<h2>Go to <a href="/video_feed">/video_feed</a> to view the webcam feed</h2>'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
