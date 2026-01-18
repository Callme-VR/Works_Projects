import os 
import cv2
import time 
import uuid
import numpy as np
import math 


image_path="Sign_data"
labels=['Hello','Yes','No','Thank You','I Love You','Please']

number_of_images=15

# Hand detection parameters
offset=20
image_size=300

def detect_hand(frame):
    """Simple hand detection using skin color"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assuming it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 1000:  # Minimum area threshold
            return True, largest_contour
    
    return False, None

def crop_and_resize_hand(frame, contour):
    """Crop and resize hand region to standard size"""
    x, y, w, h = cv2.boundingRect(contour)
    
    # Ensure crop bounds are within image
    img_h, img_w = frame.shape[:2]
    y_start = max(0, y-offset)
    y_end = min(img_h, y+h+offset)
    x_start = max(0, x-offset)
    x_end = min(img_w, x+w+offset)
    
    image_crop = frame[y_start:y_end, x_start:x_end]
    
    if image_crop.size == 0:
        return None
    
    # Create white background
    image_white = np.ones((image_size, image_size, 3), np.uint8) * 255
    
    # Resize while maintaining aspect ratio
    aspect_ratio = h/w if w > 0 else 1
    
    if aspect_ratio > 1:
        k = image_size/h
        w_cal = math.ceil(k*w)
        image_resize = cv2.resize(image_crop, (w_cal, image_size))
        w_gap = math.ceil((image_size-w_cal)/2)
        image_white[:, w_gap:w_gap+w_cal] = image_resize
    else:
        k = image_size/w
        h_cal = math.ceil(k*h)
        image_resize = cv2.resize(image_crop, (image_size, h_cal))
        h_gap = math.ceil((image_size-h_cal)/2)
        image_white[h_gap:h_gap+h_cal, :] = image_resize
    
    return image_white


for label in labels:
     image_Dir = os.path.join(image_path, label)
     os.makedirs(image_Dir, exist_ok=True)
     
     cap = cv2.VideoCapture(0)
     
     print(f"Collecting the Images for Label: {label}")
     print("Show your hand gesture to the camera. Images will be captured automatically when hand is detected.")
     time.sleep(3)
     
     images_collected = 0
     last_capture_time = 0
     
     while images_collected < number_of_images:
          ret, frame = cap.read()
          if not ret:
               print("Failed to capture image")
               continue
          
          # Detect hand in frame
          hand_detected, contour = detect_hand(frame)
          
          if hand_detected:
               # Draw bounding box around detected hand
               x, y, w, h = cv2.boundingRect(contour)
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
               cv2.putText(frame, "Hand Detected - Capturing...", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
               
               # Capture image with delay to avoid duplicates
               current_time = time.time()
               if current_time - last_capture_time > 1.0:  # 1 second delay between captures
                    # Crop and resize hand region
                    processed_hand = crop_and_resize_hand(frame, contour)
                    
                    if processed_hand is not None:
                         image_name = os.path.join(image_Dir, f"{label}_{uuid.uuid1()}.jpg")
                         cv2.imwrite(image_name, processed_hand)
                         images_collected += 1
                         last_capture_time = current_time
                         print(f"Collected image {images_collected}/{number_of_images} for {label}")
                         
                         # Show the processed hand
                         cv2.imshow('Processed Hand', processed_hand)
          else:
               cv2.putText(frame, "No Hand Detected - Show your hand", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          
          cv2.imshow('Live Feed', frame)
          
          # Allow manual quit
          if cv2.waitKey(1) & 0xFF == ord('q'):
               print("Quitting collection...")
               break
     
     cap.release()
     cv2.destroyAllWindows()
     print(f"Finished collecting {images_collected} images for {label}")
     print("Moving to next label in 3 seconds...")
     time.sleep(3)
          


