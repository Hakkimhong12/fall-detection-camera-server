import argparse
import logging
import time
import cv2
import numpy as np
import datetime
from ultralytics import YOLO
import os
from collections import deque
from multiprocessing import Process
from threading import Thread
from api import save_video_segment, save_image, MAX_FPS, CAMERA_ID
from queue import Queue


logger = logging.getLogger('FallDetection-YOLO')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

MAX_FPS = 11

BUFFER_SIZE =  5 * MAX_FPS  
FINAL_BUFFER_SIZE = 10 * MAX_FPS  

# Initialize buffers
buffer_A = deque(maxlen=BUFFER_SIZE)
buffer_B = deque(maxlen=BUFFER_SIZE)
buffer_C = deque(maxlen=BUFFER_SIZE)
buffer_D = deque(maxlen=BUFFER_SIZE)
final_buffer = deque(maxlen=FINAL_BUFFER_SIZE)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def save_image_process(frame_with_boxes, fall_detected_time, queue, camera_id):
    try:
        image_url = save_image(frame_with_boxes, fall_detected_time, camera_id)
        queue.put(image_url)
    except Exception as e:
        logger.error(f"Error in save_image_process: {str(e)}")
        queue.put(None)



def save_video_process(frames_before, before_timestamps, frames_after, after_timestamps, fall_detected_time, image_url):
    try:
        save_video_segment(
            frames_before, before_timestamps, 
            frames_after, after_timestamps, 
            fall_detected_time, image_url, 'AZ202', 640, 480
        )
    except Exception as e:
        logger.error(f"Error in save_video_process: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='YOLO-based Fall Detection')
    parser.add_argument('--video', type=str, help='path to input video file')
    parser.add_argument('--camera', type=int, default=0, help='camera device number')
    args = parser.parse_args()

    logger.debug('Initializing YOLO model')
    model = YOLO('model/fall detection model.pt')

    if args.video:
        cam = cv2.VideoCapture(args.video)
    else:
        cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        exit()

    ret_val, frame = cam.read()
    if not ret_val or frame is None:
        exit()

    logger.info('Video frame=%dx%d' % (frame.shape[1], frame.shape[0]))

    fall_start_time = None
    fall_confirmed = False
    fall_end_time = None 
    confidence_threshold = 0.50
    fall_detected = False 
    image_thread = None
    video_thread = None
    image_url = None
    EVENT = 0
    fps_time = 0
    confirmed_frame = None  

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        current_time = time.time()

        results = model(frame)
        fall_detected = False  
        frame_with_boxes = frame.copy()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = box.conf.item()

                if score < confidence_threshold:
                    continue  

                bbox = box.xyxy[0].tolist()
                class_name = model.names[class_id]

                if class_id == 0:  # falling
                    color = (0, 0, 255)  # Red for falling
                    fall_detected = True
                elif class_id == 1:  # sitting
                    color = (255, 255, 0)  # Cyan for sitting
                elif class_id == 2:  # standing
                    color = (0, 255, 0)  # Green for standing
                else:  # walking
                    color = (255, 0, 0)  # Blue for walking

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name}: {score:.2f}'
                cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

      
        buffer_B.append((time.time(), frame.copy(), frame_with_boxes))
        if len(buffer_B) == BUFFER_SIZE:
            buffer_A.append(buffer_B[0])

        if fall_detected and EVENT != 1:
            if fall_start_time is None:
                fall_start_time = time.time()
            elif time.time() - fall_start_time >= 3:  
                if not fall_confirmed:
                    print("Fall confirmed!")
                    EVENT = 1
                    confirmed_frame = frame_with_boxes.copy()  
                    cv2.putText(confirmed_frame, "FALL CONFIRMED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    fall_confirmed = True
                    fall_detected_time = datetime.datetime.now()

                    # Start the save image thread
                    image_queue = Queue()
                    image_thread = Thread(target=save_image_process, args=(confirmed_frame, fall_detected_time, image_queue, CAMERA_ID))
                    image_thread.start()
                    image_thread.join()
                    image_url = image_queue.get()

                    buffer_C.clear()
                    buffer_C.extend(list(buffer_A))
                    buffer_C.extend(list(buffer_B))

                    buffer_D.clear()
        else:
            fall_start_time = None

        if fall_confirmed:
            buffer_D.append((time.time(), frame.copy(), frame_with_boxes))
            if len(buffer_D) == BUFFER_SIZE:
                fall_end_time = time.time()

                # Wait for the image saving thread to finish before starting the video saving thread
                if image_thread:
                    image_thread.join()

                # Start the save video thread
                video_thread = Thread(target=save_video_process, args=(
                    [frame_with_boxes for _, _, frame_with_boxes in list(buffer_C)],
                    [ts for ts, _, _ in list(buffer_C)],
                    [frame_with_boxes for _, _, frame_with_boxes in list(buffer_D)],
                    [ts for ts, _, _ in list(buffer_D)],
                    fall_detected_time, image_url))
                video_thread.start()

                # Reset buffers and flags
                buffer_A.clear()
                buffer_B.clear()
                buffer_C.clear()
                buffer_D.clear()
                fall_confirmed = False 
                fall_end_time = None

        fps = 1 / (time.time() - fps_time)
        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        fps_time = time.time()

        # Display frame and check for quit key
        cv2.imshow('Fall Detection', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Graceful shutdown of threads before exiting
    if image_thread and image_thread.is_alive():
        image_thread.join()

    if video_thread and video_thread.is_alive():
        video_thread.join()

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
