import cv2
import numpy as np
import math
import os

# Simple hand detection using skin color (same as data collection)
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
    y_start = max(0, y-20)
    y_end = min(img_h, y+h+20)
    x_start = max(0, x-20)
    x_end = min(img_w, x+w+20)
    
    image_crop = frame[y_start:y_end, x_start:x_end]
    
    if image_crop.size == 0:
        return None
    
    # Create white background
    image_white = np.ones((300, 300, 3), np.uint8) * 255
    
    # Resize while maintaining aspect ratio
    aspect_ratio = h/w if w > 0 else 1
    
    if aspect_ratio > 1:
        k = 300/h
        w_cal = math.ceil(k*w)
        image_resize = cv2.resize(image_crop, (w_cal, 300))
        w_gap = math.ceil((300-w_cal)/2)
        image_white[:, w_gap:w_gap+w_cal] = image_resize
    else:
        k = 300/w
        h_cal = math.ceil(k*h)
        image_resize = cv2.resize(image_crop, (300, h_cal))
        h_gap = math.ceil((300-h_cal)/2)
        image_white[h_gap:h_gap+h_cal, :] = image_resize
    
    return image_white

# Simple placeholder classifier (replace with actual model)
class SimpleClassifier:
    def __init__(self, model_path, labels_path):
        self.labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]
        print("Note: Using placeholder classifier. Replace with actual model.")
    
    def getPrediction(self, img, draw=True):
        # Placeholder prediction - replace with actual model inference
        import random
        index = random.randint(0, len(self.labels)-1)
        prediction = [0.1] * len(self.labels)
        prediction[index] = 0.9  # Mock confidence
        return prediction, index

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300

# Initialize classifier (placeholder - replace with actual model)
try:
    classifier = SimpleClassifier("Model/keras_model.h5", "Model/labels.txt")
except:
    print("Model files not found. Using placeholder classifier.")
    classifier = SimpleClassifier("", "")

labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]

while True:
    success, img = cap.read()
    if not success:
        continue
    
    imgOutput = img.copy()
    
    # Use same hand detection as data collection
    hand_detected, contour = detect_hand(img)
    
    if hand_detected:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop and resize hand using same function as data collection
        imgWhite = crop_and_resize_hand(img, contour)
        
        if imgWhite is not None:
            # Get prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {labels[index]}, Confidence: {max(prediction):.2f}")
            
            # Draw bounding box and label
            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0,255,0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)
            
            # Show processed images
            cv2.imshow('ImageWhite', imgWhite)
            
            # Show crop region
            y_start = max(0, y-offset)
            y_end = min(img.shape[0], y+h+offset)
            x_start = max(0, x-offset)
            x_end = min(img.shape[1], x+w+offset)
            imgCrop = img[y_start:y_end, x_start:x_end]
            cv2.imshow('ImageCrop', imgCrop)
    else:
        cv2.putText(imgOutput, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Image', imgOutput)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()