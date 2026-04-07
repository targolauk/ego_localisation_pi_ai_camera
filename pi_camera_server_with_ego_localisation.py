import argparse
import sys
import time
from functools import lru_cache
import csv
import os
import threading
import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (
    NetworkIntrinsics,
    postprocess_nanodet_detection
)
from collections import deque
from scipy.optimize import least_squares
import select
import tty
import termios


# --- Setup for non-blocking key read ---
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

# ===============================================
# CONFIG
# ===============================================
traffic_signs = {
    "A": {"coords": np.array([0.0, 0.0])},
    "B": {"coords": np.array([50.0, 0.0])},
    "C": {"coords": np.array([25.0, 40.0])},
}

KNOWN_SIZE_CM = 60.0
KNOWN_DISTANCE_M = 5.0
MEMORY_TIME = 10

SIGN_MEMORY = deque(maxlen=50)

last_detections = []

trajectory_history = deque(maxlen=200)

focal_length = None
calibrated = False

# ===============================================
# TRILATERATION
# ===============================================
def estimate_position(signs_data):
    positions = [s["coords"] for s in signs_data]
    distances = [s["dist"] for s in signs_data]

    initial = np.mean(positions, axis=0)

    res = least_squares(
        lambda p: [np.linalg.norm(p - pos) - d for pos, d in zip(positions, distances)],
        initial,
        bounds=([-500, -500], [500, 500])
    )
    return res.x, res.cost

# ===============================================
# UPDATE MEMORY + TRACKING (with SORT)
# ===============================================

def update_sign_memory(detections, frame_width, tracked_objects=None):
    """Append each tracked object as a new sign if its ID is new."""
    global SIGN_MEMORY
    current_time = time.time()

    if tracked_objects is None:
        tracked_objects = []

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        w, h = x2 - x1, y2 - y1
        cx = x1 + w / 2

        # Map horizontal sector to sign ID
        if cx < frame_width / 3:
            sign_id = "A"
        elif cx < 2 * frame_width / 3:
            sign_id = "B"
        else:
            sign_id = "C"

        # Estimate distance
        size_px = max(w, h)
        distance_m = (KNOWN_SIZE_CM / 100) * focal_length / size_px if calibrated else None

        # Only add if this track_id is not already in SIGN_MEMORY
        if track_id not in [s["track_id"] for s in SIGN_MEMORY]:
            SIGN_MEMORY.append({
                "track_id": track_id,
                "id": f"{sign_id}_{int(track_id)}",
                "coords": traffic_signs[sign_id]["coords"],
                "dist": distance_m,
                "time": current_time
            })

    # Remove old entries (>MEMORY_TIME)
    SIGN_MEMORY = deque(
        [s for s in SIGN_MEMORY if current_time - s["time"] <= MEMORY_TIME],
        maxlen=50
    )

def update_memory_with_sort(detections, tracker, frame_width):
    """
    Update SORT tracker and SIGN_MEMORY.
    Only adds new memory entries when SORT generates a new track_id.
    """
    if not detections:
        return

    # Prepare SORT input: [x1, y1, x2, y2, score]
    dets_for_sort = []
    for det in detections:
        x, y, w, h = det.box
        dets_for_sort.append([x, y, x + w, y + h, det.conf])

    # Update tracker
    tracked_objects = tracker.update(np.array(dets_for_sort)) if dets_for_sort else []


    # Update SIGN_MEMORY using tracked IDs
    update_sign_memory(detections, frame_width, tracked_objects)
# ===============================================
# DETECTION CLASS
# ===============================================
class Detection:
    def __init__(self, coords, category, conf):
        self.category = category
        self.conf = conf

        x1, y1, x2, y2 = coords

        x_scale, y_scale = 1920 / 640, 1080 / 640
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)

        self.box = (x1, y1, x2 - x1, y2 - y1)
        self.distance_m = None

# ===============================================
# PARSE DETECTIONS (FIXED)
# ===============================================
def parse_detections(metadata: dict):
    global last_detections, focal_length, calibrated

    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    # Fetch raw outputs from the network
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()

    # Debug: print shapes of raw outputs
    if np_outputs is None:

        return last_detections

    # --- Get raw boxes ---
    if intrinsics.postprocess == "nanodet":

        try:
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0],
                conf=threshold,
                iou_thres=iou,
                max_out_dets=max_detections
            )[0]

            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)

        except Exception as e:

            return last_detections

    else:
        # Extract boxes, scores, classes
        boxes_raw, scores_raw, classes_raw = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        # Normalize boxes if required
        if bbox_normalization:
            boxes_raw = boxes_raw / input_h

        # Reorder boxes if required
        if bbox_order == "xy":
            boxes_raw = boxes_raw[:, [1, 0, 3, 2]]

        # Flatten boxes into list of tuples (x1, y1, x2, y2) as floats
        boxes = []
        for b in boxes_raw:
            boxes.append([float(coord) for coord in b])

        scores = [float(s) for s in scores_raw]
        classes = [int(c) for c in classes_raw]

    detections = []
    for box, score, category in zip(boxes, scores, classes):
        if score <= threshold:
            continue

        # FIXED: pass only expected args to Detection
        det = Detection(box, category, score)

        # =========================
        # DISTANCE ESTIMATION
        # =========================
        x, y, w, h = det.box
        size_px = max(w, h)

        # --- Auto-calibrate focal length ---
        if not calibrated and size_px > 50:
            focal_length = (size_px * KNOWN_DISTANCE_M) / (KNOWN_SIZE_CM / 100)
            calibrated = True

        # --- Estimate distance ---
        if calibrated and size_px > 30:
            det.distance_m = (KNOWN_SIZE_CM / 100) * focal_length / size_px
        else:
            det.distance_m = None

        detections.append(det)

    last_detections = detections
    return last_detections

# ===============================================
# DRAW
# ===============================================
def draw_detections(request, stream="main"):
    global last_detections

    if last_detections is None:
        return

    with MappedArray(request, stream) as m:
        for d in last_detections:
            x, y, w, h = d.box

            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if d.distance_m:
                label = f"{d.distance_m:.2f}m"
                cv2.putText(m.array, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# ===============================================
# ARGS
# ===============================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="network.rpk")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Initialize IMX500 and intrinsics
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.labels = ["sign"]
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    picam2.pre_callback = draw_detections
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080)},
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12
    )
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    # Initialize SORT tracker
    from sort import Sort
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

    try:
        while True:
            # --- 1. Parse detections ---
            last_results = parse_detections(picam2.capture_metadata())
            print(f"[INFO] {len(last_results)} detections")

            # --- 2. Update memory with SORT ---
            update_memory_with_sort(last_results, tracker, frame_width=1920)
            print("Sign memory:", len(SIGN_MEMORY))
            # --- 3. Ego-localization ---
            if len(SIGN_MEMORY) >= 3 and calibrated:
                try:
                    ego_pos, cost = estimate_position(list(SIGN_MEMORY))  # just pass the deque as list
                    trajectory_history.append((ego_pos[0], ego_pos[1]))
                    print(f"[LOCALIZATION] Position ≈ ({ego_pos[0]:.2f}, {ego_pos[1]:.2f}) m | error={cost:.4f}")
                except Exception as e:
                    print("Trilateration error:", e)

            # --- 4. Capture + save frame if 's' pressed ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                timestamp = f"{int(time.time() * 1000)}"
                request = picam2.capture_request()
                with MappedArray(request, "main") as m:
                    frame = cv2.cvtColor(m.array.copy(), cv2.COLOR_RGB2BGR)
                    filename = f"video_data/frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[INFO] Saved frame {filename}")
                request.release()

    except KeyboardInterrupt:
        picam2.stop()
        print("Stopped")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        picam2.stop()
        print("Stopped")
