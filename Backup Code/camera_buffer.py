from flask import Flask, Response, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import time
import datetime
from ultralytics import YOLO
from collections import deque
from threading import Thread
from queue import Queue
import logging
import torch
from api import (
    save_video_segment_buffer, save_image, get_fall_detection_record_time, 
    get_video_footage_record_time, CAMERA_ID, MAX_FPS
)

app = Flask(__name__)
socketio = SocketIO(app)

# Logger setup
logger = logging.getLogger('FallDetection-YOLO')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Global variables
TIME = get_fall_detection_record_time(CAMERA_ID)
if TIME is None:
    logger.error("Failed to fetch fallDetectionRecordTime. Exiting.")
    exit()

BUFFER_SIZE = TIME * MAX_FPS
FINAL_BUFFER_SIZE = 2 * TIME * MAX_FPS

FOOTAGE_TIME = get_video_footage_record_time(CAMERA_ID)
if FOOTAGE_TIME is None:
    logger.error("Failed to fetch videoFootageRecordTime. Exiting.")
    exit()

# Initialize buffers
buffer_A = deque(maxlen=BUFFER_SIZE)
buffer_B = deque(maxlen=BUFFER_SIZE)
buffer_C = deque(maxlen=BUFFER_SIZE)
buffer_D = deque(maxlen=BUFFER_SIZE)
final_buffer = deque(maxlen=FINAL_BUFFER_SIZE)

# Thread management
image_thread = None
video_thread = None
image_queue = Queue()

# YOLO model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('model/fall detection model.pt').to(device)

def save_image_process(frame_with_boxes, fall_detected_time, queue, camera_id):
    try:
        image_url = save_image(frame_with_boxes, fall_detected_time, camera_id)
        queue.put(image_url)
    except Exception as e:
        logger.error(f"Error in save_image_process: {str(e)}")
        queue.put(None)

def save_video_process(frames_before, before_timestamps, frames_after, after_timestamps, fall_detected_time, image_url):
    try:
        save_video_segment_buffer(
            frames_before, before_timestamps, 
            frames_after, after_timestamps, 
            fall_detected_time, image_url, CAMERA_ID, 640, 480
        )
    except Exception as e:
        logger.error(f"Error in save_video_process: {str(e)}")

def process_frame(frame):
    global image_thread, video_thread, buffer_A, buffer_B, buffer_C, buffer_D

    current_time = time.time()
    results = model(frame)
    fall_detected = False
    frame_with_boxes = frame.copy()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = box.conf.item()

            if score < 0.50:
                continue

            bbox = box.xyxy[0].tolist()
            class_name = model.names[class_id]

            if class_id == 0:  # falling
                color = (0, 0, 255)
                fall_detected = True
            elif class_id == 1:  # sitting
                color = (255, 255, 0)
            elif class_id == 2:  # standing
                color = (0, 255, 0)
            else:  # walking
                color = (255, 0, 0)

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name}: {score:.2f}'
            cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    buffer_B.append((current_time, frame.copy(), frame_with_boxes))
    if len(buffer_B) == BUFFER_SIZE:
        buffer_A.append(buffer_B[0])

    return frame_with_boxes, fall_detected

def gen_frames():
    global image_thread, video_thread, buffer_A, buffer_B, buffer_C, buffer_D

    cam = cv2.VideoCapture(0)  # Use camera index 0, adjust if needed
    fall_start_time = None
    fall_confirmed = False
    EVENT = 0

    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frame_with_boxes, fall_detected = process_frame(frame)

            if fall_detected and not fall_confirmed and EVENT != 1:
                if fall_start_time is None:
                    fall_start_time = time.time()
                elif time.time() - fall_start_time >= 3:
                    print("Fall confirmed!")
                    EVENT = 1
                    fall_confirmed = True
                    fall_detected_time = datetime.datetime.now()

                    if image_thread is None or not image_thread.is_alive():
                        image_thread = Thread(target=save_image_process, args=(frame_with_boxes, fall_detected_time, image_queue,CAMERA_ID))
                        image_thread.start()

                    buffer_C.clear()
                    buffer_C.extend(list(buffer_A))
                    buffer_C.extend(list(buffer_B))
                    buffer_D.clear()
            else:
                fall_start_time = None

            if fall_confirmed:
                buffer_D.append((time.time(), frame.copy(), frame_with_boxes))
                if len(buffer_D) == BUFFER_SIZE:
                    if image_thread and image_thread.is_alive():
                        image_thread.join()

                    try:
                        image_url = image_queue.get_nowait()
                    except Queue.Empty:
                        logger.error("Failed to get image URL from queue")
                        image_url = None

                    if video_thread is None or not video_thread.is_alive():
                        video_thread = Thread(target=save_video_process, args=(
                            [frame_with_boxes for _, _, frame_with_boxes in list(buffer_C)],
                            [ts for ts, _, _ in list(buffer_C)],
                            [frame_with_boxes for _, _, frame_with_boxes in list(buffer_D)],
                            [ts for ts, _, _ in list(buffer_D)],
                            fall_detected_time, image_url))
                        video_thread.start()

                    buffer_A.clear()
                    buffer_B.clear()
                    buffer_C.clear()
                    buffer_D.clear()
                    fall_confirmed = False

            ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fall_streaming')
def fall_streaming():
    return render_template('fall_streaming.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)