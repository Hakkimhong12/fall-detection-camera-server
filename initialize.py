import torch
import logging
from ultralytics import YOLO
import logging
import os
from api import get_fall_detection_record_time, get_video_footage_record_time
from config import MAX_FPS

def initialize_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def initialize_model(model_path='model/fall detection model.pt'):
    logger = initialize_logging()
    try:
        # Determine the device to use (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        model = YOLO(model_path).to(device)
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        return model, device

    except Exception as e:
        logger.error(f"Error initializing the model: {str(e)}")
        return None, None

def initialize_size(CAMERA_ID):
    logger = initialize_logging()
    
    TIME = get_fall_detection_record_time(CAMERA_ID)
    logger.info(f"Fall Segment Video Duration: {TIME} seconds")
    if TIME is None:
        logger.error("Failed to fetch fallDetectionRecordTime. Exiting.")
        exit()
    BUFFER_SIZE = TIME * MAX_FPS
    FINAL_BUFFER_SIZE = 2 * TIME * MAX_FPS
    FOOTAGE_TIME = get_video_footage_record_time(CAMERA_ID)
    logger.info(f"Video Footage Duration: {FOOTAGE_TIME} seconds\n")
    if FOOTAGE_TIME is None:
        logger.error("Failed to fetch videoFootageRecordTime. Exiting.")
        exit()
    
    return BUFFER_SIZE, FINAL_BUFFER_SIZE, FOOTAGE_TIME, TIME

def result_footage_time(CAMERA_ID):
    _, value = get_video_footage_record_time(CAMERA_ID)
    return value
