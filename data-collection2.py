import os 
import cv2
import time 
import uuid
import numpy as np
import math 


image_path = "Face_data"
labels = ['smile', 'angry', 'normal','sad']

number_of_images = 15

offset = 20
image_size = 300

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_face(frame):
    """Detect face using Haar cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) > 0:
        # Return the largest face
        areas = [w * h for (x, y, w, h) in faces]
        max_idx = np.argmax(areas)
        return True, faces[max_idx]
    
    return False, None


def crop_and_resize_face(frame, face_coords):
    """Crop and resize face region to standard size"""
    x, y, w, h = face_coords
    
    img_h, img_w = frame.shape[:2]
    y_start = max(0, y - offset)
    y_end = min(img_h, y + h + offset)
    x_start = max(0, x - offset)
    x_end = min(img_w, x + w + offset)
    
    image_crop = frame[y_start:y_end, x_start:x_end]
    
    if image_crop.size == 0:
        return None
    
    image_white = np.ones((image_size, image_size, 3), np.uint8) * 255
    
    aspect_ratio = h / w if w > 0 else 1
    
    if aspect_ratio > 1:
        k = image_size / h
        w_cal = math.ceil(k * w)
        image_resize = cv2.resize(image_crop, (w_cal, image_size))
        w_gap = math.ceil((image_size - w_cal) / 2)
        image_white[:, w_gap:w_gap + w_cal] = image_resize
    else:
        k = image_size / w
        h_cal = math.ceil(k * h)
        image_resize = cv2.resize(image_crop, (image_size, h_cal))
        h_gap = math.ceil((image_size - h_cal) / 2)
        image_white[h_gap:h_gap + h_cal, :] = image_resize
    
    return image_white


for label in labels:
    image_Dir = os.path.join(image_path, label)
    os.makedirs(image_Dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    print(f"Collecting the Images for Label: {label}")
    print(f"Show your face with '{label}' expression to the camera. Images will be captured automatically when face is detected.")
    time.sleep(3)
    
    images_collected = 0
    last_capture_time = 0
    
    
    while images_collected < number_of_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue
        
        face_detected, face_coords = detect_face(frame)
        
        if face_detected:
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face Detected - Capturing '{label}'...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_time = time.time()
            if current_time - last_capture_time > 1.0:
                processed_face = crop_and_resize_face(frame, face_coords)
                
                if processed_face is not None:
                    image_name = os.path.join(image_Dir, f"{label}_{uuid.uuid1()}.jpg")
                    cv2.imwrite(image_name, processed_face)
                    images_collected += 1
                    last_capture_time = current_time
                    print(f"Collected image {images_collected}/{number_of_images} for {label}")
                    
                    cv2.imshow('Processed Face', processed_face)
        else:
            cv2.putText(frame, "No Face Detected - Show your face", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Live Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting collection...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting {images_collected} images for {label}")
    print("Moving to next label in 3 seconds...")
    time.sleep(3)
