from flask import Flask, Response, render_template
from flask_socketio import SocketIO
import cv2
import time
import datetime
from datetime import datetime
from collections import deque
from threading import Thread, Event
import threading
from queue import Queue, Empty
from initialize import initialize_model, initialize_logging, initialize_size, result_footage_time
from config import FRAME_WIDTH, FRAME_HEIGHT, MAX_FPS
from api import save_image, save_video_segment_buffer, upload_video_footage, get_camera_id, send_email, upload_notification_video
CAMERA_ID = get_camera_id()

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize
logger = initialize_logging()
model, device = initialize_model()

# Fall detection variables
last_fall_confirmation_time = None
fall_detection_enabled = True

# Initialize buffers
BUFFER_SIZE, FINAL_BUFFER_SIZE, FOOTAGE_TIME, TIME = initialize_size(CAMERA_ID)
buffer_A = deque(maxlen=BUFFER_SIZE)
buffer_B = deque(maxlen=BUFFER_SIZE)
buffer_C = deque(maxlen=BUFFER_SIZE)
buffer_D = deque(maxlen=BUFFER_SIZE)
final_buffer = deque(maxlen=FINAL_BUFFER_SIZE)

# Thread management
image_thread = None
footage_video_thread = None
image_queue = Queue()
stop_thread = Event()
current_video_writer = None 
current_video_start_time = None 

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
        
def save_notification_video(buffer_C, fall_detected_time, camera_id):
    try:
        filename = f"notification_{camera_id}_{fall_detected_time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(filename, fourcc, MAX_FPS, (FRAME_WIDTH, FRAME_HEIGHT))

        for _, _, frame in buffer_C:
            video_writer.write(frame)

        video_writer.release()
        return filename
    except Exception as e:
        logger.error(f"Error saving notification video: {str(e)}")
        return None


def save_and_upload_video_footage():
    global current_video_writer, current_video_start_time
    
    if current_video_writer:
        logger.info('Finalizing current video segment')
        current_video_writer.release()
        filename = f"{CAMERA_ID}_{datetime.fromtimestamp(current_video_start_time).strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        threading.Thread(target=upload_video_footage, args=(filename,CAMERA_ID)).start()  # Upload in a separate thread
        current_video_writer = None
        current_video_start_time = None

def process_frame(frame):
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

def video_saving_thread():
    global current_video_writer, current_video_start_time
    
    while not stop_thread.is_set():
        current_time = time.time()
        
        if current_video_start_time is None or (current_time - current_video_start_time) >= FOOTAGE_TIME:
            save_and_upload_video_footage()
            
            current_video_start_time = current_time
            filename = f"{CAMERA_ID}_{datetime.fromtimestamp(current_video_start_time).strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            current_video_writer = cv2.VideoWriter(filename, fourcc, MAX_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            logger.info(f'Started new video segment: {filename}')
        
        time.sleep(1)  # Check every second

    logger.info('Stopping video saving thread')
    save_and_upload_video_footage()  # Final save when stopping

def gen_frames():
    global image_thread, footage_video_thread, buffer_A, buffer_B, buffer_C, buffer_D
    global current_video_writer, last_fall_confirmation_time, fall_detection_enabled

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        logger.error('Error opening video capture')
        return
    logger.info('Camera opened successfully')
    fall_start_time = None
    fall_confirmed = False
    EVENT = 0
    fps_time =0
    
    if footage_video_thread is None or not footage_video_thread.is_alive():
        footage_video_thread = threading.Thread(target=video_saving_thread)
        footage_video_thread.start()
        logger.info('Video saving thread started')

    while True:
        success, frame = cam.read()
        if not success:
            logger.error('Failed to read frame from camera')
            break
        else:
            frame_with_boxes, fall_detected = process_frame(frame)
            current_time = time.time()
            
            if current_video_writer:
                current_video_writer.write(frame)

            # Check if fall detection is enabled
            if fall_detection_enabled:
                if fall_detected and not fall_confirmed and EVENT == 0:
                    if fall_start_time is None:
                        fall_start_time = current_time
                    elif current_time - fall_start_time >= 3:
                        logger.info("Fall confirmed!")
                        EVENT = 1
                        fall_confirmed = True
                        fall_detected_time = datetime.now()
                        last_fall_confirmation_time = current_time

                        if image_thread is None or not image_thread.is_alive():
                            image_thread = Thread(target=save_image_process, args=(frame_with_boxes, fall_detected_time, image_queue, CAMERA_ID))
                            image_thread.start()

                        buffer_C.clear()
                        buffer_C.extend(list(buffer_A))
                        buffer_C.extend(list(buffer_B))
                        buffer_D.clear()

                        # Disable fall detection after confirming a fall
                        logger.info(f"Fall detections system reset for : {TIME/60} minutes.")
                        fall_detection_enabled = False
                else:
                    fall_start_time = None
            else:
                # Check if it's time to re-enable fall detection
                if current_time - last_fall_confirmation_time > TIME:
                    fall_detection_enabled = True
                    EVENT = 0  
                    logger.info(f"Fall detection system starting after reset {TIME/60} minutes...")

            if fall_confirmed:
                fall_frame = frame_with_boxes
                notification_video_filename = save_notification_video(buffer_C, fall_detected_time, CAMERA_ID)
                if notification_video_filename:
                    notification_video_url = upload_notification_video(notification_video_filename, CAMERA_ID)
                    if notification_video_url:
                        send_email(notification_video_url, fall_detected_time, fall_frame, CAMERA_ID)
                buffer_D.append((time.time(), frame.copy(), frame_with_boxes))
                if len(buffer_D) == BUFFER_SIZE:
                    if image_thread and image_thread.is_alive():
                        image_thread.join()

                    try:
                        image_url = image_queue.get(timeout=5)
                    except Empty:
                        logger.error("Failed to get image URL from queue")
                        image_url = None

                    fall_video_thread = Thread(target=save_video_process, args=(
                            [frame_with_boxes for _, _, frame_with_boxes in list(buffer_C)],
                            [ts for ts, _, _ in list(buffer_C)],
                            [frame_with_boxes for _, _, frame_with_boxes in list(buffer_D)],
                            [ts for ts, _, _ in list(buffer_D)],
                            fall_detected_time, image_url))
                    fall_video_thread.start()

                    buffer_A.clear()
                    buffer_B.clear()
                    buffer_C.clear()
                    buffer_D.clear()
                    fall_confirmed = False
                    # We don't reset EVENT here because fall detection is still disabled
            fps = 1 / (time.time() - fps_time)
            cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            fps_time = time.time()

            ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()
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
    try:
        socketio.run(app, debug=False)      
    except Exception as e:
        logger.error(f'Error running the flask app: {str(e)}')
    finally:
        stop_thread.set()
        logger.info('Stop signal set, waiting for thread to finish')
        if footage_video_thread:
            footage_video_thread.join()
        logger.info('Flask app stopped!')