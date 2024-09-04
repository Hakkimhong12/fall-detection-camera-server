import cv2
import os 
from datetime import datetime as dt 
import ssl 
import smtplib
from datetime import datetime 
from email.message import EmailMessage
#from config import CAMERA_ID
import firebase_admin
from firebase_admin import credentials, storage, firestore
from prettytable import PrettyTable
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.cloud.firestore_v1.base_collection')
# Firebase initialization
key_path = os.getenv('FALL_DETECTION_PROJECT')
if key_path is None:
    raise EnvironmentError("FALL_DETECTION_PROJECT environment variable is not set.")

cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'falldetectionwebsite.appspot.com'
})
bucket = storage.bucket()
db = firestore.client()
current_video_writer = None 

def fetch_camera_info():
    camera_docs = db.collection('Patient Informations').order_by('timestamp', direction=firestore.Query.DESCENDING).get()
    camera_info = []
    for doc in camera_docs:
        camera_data = doc.to_dict()
        if 'cameraId' in camera_data and 'patientName' in camera_data and 'timestamp' in camera_data:
            camera_id = camera_data['cameraId']
            patient_name = camera_data['patientName']
            timestamp = camera_data['timestamp']
            camera_info.append((camera_id, patient_name, timestamp))
    return camera_info

def display_camera_table(camera_info):
    table = PrettyTable()
    table.field_names = ["Order", "Camera ID", "Patient Name", "Timestamp"]
    for i, (camera_id, patient_name, timestamp) in enumerate(camera_info, 1):
        table.add_row([i, camera_id, patient_name, timestamp.strftime('%Y-%m-%d %H:%M:%S')])
    print(table)

def select_camera():
    camera_info = fetch_camera_info()
    display_camera_table(camera_info)
    
    while True:
        selection = input("Enter the exact camera ID you want to use: ").strip()
        if selection in [info[0] for info in camera_info]:
            return selection
        else:
            print("Invalid camera ID. Please try again.")

def get_camera_id():
    return select_camera()

def get_camera_details(camera_id):
    try:
        camera_ref = db.collection('Patient Informations')
        query = camera_ref.where('cameraId', '==', camera_id).limit(1)
        results = query.stream()
        
        camera_details = None 
        for result in results:
            camera_details = result.to_dict()
        
        if camera_details is None:
            print(f"No details found for camera ID: {camera_id}")
            return None
        
        location = camera_details['room']
        return location
    except KeyError as e:
        print(f"Missing expected data: {str(e)}")
        return None
    except Exception as e:
        print(f"Error fetching camera details: {str(e)}")
        return None
    
# Get user's email from specific CAMERA ID
def get_users_by_camera_id(camera_id):
    try:
        # Retrieve userUids from 'Camera Access' using cameraId
        camera_access_ref = db.collection('Camera Access')
        access_query = camera_access_ref.where('cameraId', '==', camera_id)
        access_docs = access_query.stream()
        
        user_uids = [doc.to_dict().get('userUid') for doc in access_docs]
        
        if not user_uids:
            print(f"No users found with camera ID: {camera_id}")
            return None

        # Retrieve emails from 'User Information' using userUids
        email_receivers = []
        user_ref = db.collection('User Informations')
        for user_uid in user_uids:
            user_doc = user_ref.document(user_uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                email = user_data.get('email')
                if email:
                    email_receivers.append(email)
        
        return email_receivers

    except Exception as e:
        print(f"Error fetching users by camera ID: {camera_id} - {str(e)}")
        return None


# Caculate the average frame per second
def calculate_avg_fps(frames_before, before_timestamps, frames_after=None, after_timestamps=None):
    if not frames_before or not before_timestamps:
        return 0  
    

    start_time = before_timestamps[0]
    end_time = after_timestamps[-1] if after_timestamps else before_timestamps[-1]
    
    total_frames = len(frames_before) + (len(frames_after) if frames_after else 0)
    total_duration = end_time - start_time
    

    if total_duration < 0.1: 
        return 0 
    
    avg_fps = total_frames / total_duration
    return avg_fps

# Upload fall image to 'Fall Image Folder' 
def save_image(frame, fall_time, camera_id):
    filename = f"{camera_id}_{fall_time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    
    if not os.path.exists(filename):
        print(f"Failed to create image file: {filename}")
        return None
    
    try:
        blob = bucket.blob(f"Fall Images Folder/{filename}")
        blob.upload_from_filename(filename, content_type='image/jpeg')
        blob.metadata = {'contentType': 'image/jpeg'}
        blob.make_public()
        
        print(f"Image uploaded to Firebase Storage: {filename}")
        
      
        os.remove(filename)
        
        return blob.public_url  # Only return the public URL
    except Exception as e:
        print(f"Error uploading image or saving to Firebase Storage: {str(e)}")
        return None

# This function use for buffer concept
def save_video_segment_buffer(frames_before, before_timestamps, frames_after, after_timestamps, fall_time, image_url, camera_id, target_width=640, target_height=480):
    from initialize import initialize_logging
    logger = initialize_logging()
    location = get_camera_details(camera_id)

    filename = f"{camera_id}_{fall_time.strftime('%Y%m%d_%H%M%S')}.mp4"

    def resize_frame(frame, target_width, target_height):
        return cv2.resize(frame, (target_width, target_height))

    resized_frames_before = [resize_frame(frame, target_width, target_height) for frame in frames_before]
    resized_frames_after = [resize_frame(frame, target_width, target_height) for frame in frames_after]

    avg_fps = calculate_avg_fps(frames_before, before_timestamps, frames_after, after_timestamps)

    if avg_fps > 0:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(filename, fourcc, avg_fps, (target_width, target_height))

            for frame in resized_frames_before:
                out.write(frame)
            for frame in resized_frames_after:
                out.write(frame)

            out.release()
            print(f"Video saved temporarily: {filename}")
            print(f"Average FPS: {avg_fps:.2f}")

            # Upload the video to Firebase
            logger.info('Uploading video segment to Firebase Storage...')
            blob = bucket.blob(f"Fall Videos Folder/{filename}")
            blob.upload_from_filename(filename, content_type='video/mp4')
            blob.metadata = {'contentType': 'video/mp4'}
            blob.make_public()
            logger.info(f"Video uploaded to Firebase Storage: {filename}")


            if os.path.exists(filename):
                os.remove(filename)


            doc_ref = db.collection('Fall Detections').document()
            doc_ref.set({
                'cameraId': camera_id,
                'timestamp': fall_time,
                'imageUrl': image_url, 
                'videoUrl': blob.public_url,
                'room': location
            })

            logger.info(f"Fall event data saved to Firestore with ID: {doc_ref.id}")
            return blob.public_url, doc_ref.id, resized_frames_before[-1], resized_frames_after[-1]

        except Exception as e:
            print(f"Error uploading video segment or saving to Firestore: {str(e)}")
            return None, None, None, None
    else:
        return None, None, None, None
    
# Get all information of Camera Setting 
def fetch_all_camera_settings():
    try:
        camera_settings_ref = db.collection('Camera Informations')
        docs = camera_settings_ref.stream()
        
        camera_settings = []
        for doc in docs:
            camera_settings.append(doc.to_dict())
        
        return camera_settings
    except Exception as e:
        print(f"Error fetching camera settings: {str(e)}")
        return None
    
# Get duration time for segment video
def get_fall_detection_record_time(camera_id):
    try:
        camera_setting_ref = db.collection('Camera Informations')
        query = camera_setting_ref.where('cameraId', '==', camera_id).limit(1)
        results = query.stream()
        
        fall_detection_record_time = None
        for result in results:
            record_time_str = result.to_dict().get('fallDetectionRecordTime')
            
            if record_time_str:
                # Extract numeric value
                record_time_value = int(''.join(filter(str.isdigit, record_time_str)))
                # Check the suffix and convert to seconds if needed
                if 'minutes' in record_time_str:
                    fall_detection_record_time = record_time_value * 60 
                elif 's' in record_time_str:
                    fall_detection_record_time = record_time_value
                elif 'mn' in record_time_str:
                    fall_detection_record_time = record_time_value * 60
        
        if fall_detection_record_time is None:
            print(f"No fallDetectionRecordTime found for camera ID: {camera_id}")
        
        return fall_detection_record_time
    except Exception as e:
        print(f"Error fetching fallDetectionRecordTime: {str(e)}")
        return None
    

# Get video footage recording time  
def get_video_footage_record_time(camera_id):
    try:
        camera_setting_ref = db.collection('Camera Informations')
        query = camera_setting_ref.where('cameraId', '==', camera_id).limit(1)
        results = query.stream()
        
        fall_detection_record_time = None
        for result in results:
            record_time_str = result.to_dict().get('cameraRecordTime')
            
            if record_time_str:
                # Extract numeric value
                record_time_value = int(''.join(filter(str.isdigit, record_time_str)))
                # Check the suffix and convert to seconds if needed
                if 'h' in record_time_str:
                    fall_detection_record_time = record_time_value * 3600
                elif 'mn' in record_time_str:
                    fall_detection_record_time = record_time_value * 60
        
        if fall_detection_record_time is None:
            print(f"No cameraRecordTime found for camera ID: {camera_id}")
        
        return fall_detection_record_time
    except Exception as e:
        print(f"Error fetching cameraRecordTime: {str(e)}")
        return None
        
# Send email to all user (CAMERA ID)
def send_email(video_url, fall_time, fall_frame, camera_id):
    location = get_camera_details(camera_id)
    if location is None: 
        print(f"Failed to fetch camera details for camera ID: {camera_id}")
        return 
    
    email_receivers = get_users_by_camera_id(camera_id)
    if not email_receivers:
        print(f"No email receivers found for camera ID: {camera_id}")
        return 
    
    tdatetime = datetime.now()
    tstr = tdatetime.strftime("%Y-%m-%d %H:%M:%S")
    email_sender = 'godapple79@gmail.com'
    email_password = 'pblh ylrc vlgi aomt'
    
    subject = 'Emergency Alert: Fall Detected'
    body = f"""
    Fall Details:
    - Detection Time: {fall_time}
    - Current Time: {tstr}
    - Camera ID: {camera_id}
    - Room: {location}
    - Fall Video URL: {video_url}
    """

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = ', '.join(email_receivers)
    em['Subject'] = subject
    em.set_content(body)
    
    _, fall_img_encoded = cv2.imencode('.jpg', fall_frame)
    # _, last_img_encoded = cv2.imencode('.jpeg', last_frame)
    
    em.add_attachment(fall_img_encoded.tobytes(), maintype='image', subtype='jpg', filename='fall_frame.jpg')
    # em.add_attachment(last_img_encoded.tobytes(), maintype='image', subtype='jpg', filename='last_frame.jpg')
    
    context = ssl.create_default_context()
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.send_message(em)
    print(f"Email sent successfully to {', '.join(email_receivers)}.")
    
def upload_video_footage(filename, CAMERA_ID):
    from initialize import initialize_logging
    logger = initialize_logging()
    logger.info(f"Uploading video: {filename}")
    try:
        blob = bucket.blob(f"Video Footage/{CAMERA_ID}/{os.path.basename(filename)}")
        blob.upload_from_filename(filename)
        blob.make_public()
        public_url = blob.public_url
        logger.info(f"Uploaded video successfully: {filename}")
        logger.info(f"Public URL: {public_url}")
        
        # Save public URL to a text file
        with open('video_urls.txt', 'a') as f:
            f.write(f"{filename}: {public_url}\n")
        
        os.remove(filename)
        logger.info(f"Deleted local file: {filename}")
    except Exception as e:
        logger.error(f"Error uploading video {filename}: {str(e)}")  

# Save Video footage
# This function use for buffer concept for save video footage but we need to move on. Use disk instead.
def save_video_footage(frames, timestamps, camera_id):
    from initialize import initialize_logging
    logger = initialize_logging()
    try:
        # Create a custom video file name with .mp4 extension
        current_time_str = datetime.datetime.now().strftime('%d-%b-%Y_%I-%M_%p')
        video_filename = f"{camera_id}_{current_time_str}.mp4"

        logger.info(f"Started processing continuous video segment: {video_filename}")

        target_width, target_height = 640, 480
        resized_frames = [cv2.resize(frame, (target_width, target_height)) for frame in frames]

        avg_fps = calculate_avg_fps(resized_frames, timestamps, [], [])

        if avg_fps > 0:
            # Use 'avc1' codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(video_filename, fourcc, avg_fps, (target_width, target_height))

            for frame in resized_frames:
                out.write(frame)

            out.release()
            logger.info(f"Video saved temporarily: {video_filename}")
            logger.info(f"Average FPS: {avg_fps:.2f}")

            # Upload the video to Firebase
            bucket = storage.bucket()
            blob = bucket.blob(f'Video Footage/{camera_id}/{video_filename}')
            blob.upload_from_filename(video_filename, content_type='video/mp4')
            blob.make_public()

            # Remove the local video file after upload
            if os.path.exists(video_filename):
                os.remove(video_filename)

            return blob.public_url

        else:
            logger.error("Average FPS is 0, skipping video creation.")
            return None

    except Exception as e:
        logger.error(f"Error in save_video_footage: {str(e)}")
        return None
    
    
def upload_notification_video(filename, camera_id):
    try:
        bucket = storage.bucket()
        storage_ref = bucket.blob(f'Notification Video/{filename}')
        storage_ref.upload_from_filename(filename)
        storage_ref.make_public()
        return storage_ref.public_url
    except Exception as e:
        print(f"Error uploading video to Firebase: {str(e)}")
        return None
