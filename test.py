import cv2
import numpy as np
import math
import os
import time
import tensorflow as tf
from collections import Counter, deque

try:
    import tf_keras as tfk  # type: ignore
except Exception:
    tfk = None

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
        self.labels = self._load_labels(labels_path)
        self.model = None
        try:
            # Many Teachable Machine / older Keras .h5 models load better with tf-keras.
            if tfk is not None:
                self.model = tfk.models.load_model(model_path, compile=False)
            else:
                self.model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print("Failed to load model. Predictions will be disabled.")
            print(e)
 
    def _load_labels(self, labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        labels.append(parts[1])
                    else:
                        labels.append(parts[0])
                return labels
        except Exception:
            return []
     
    def getPrediction(self, img, draw=True):
        if self.model is None or not self.labels:
            return [], -1
 
        input_shape = self.model.input_shape
        target_h = input_shape[1] if len(input_shape) >= 3 and input_shape[1] is not None else 300
        target_w = input_shape[2] if len(input_shape) >= 3 and input_shape[2] is not None else 300
 
        img_resized = cv2.resize(img, (target_w, target_h))
        x = img_resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = self.model.predict(x, verbose=0)[0]
        index = int(np.argmax(pred))
 
        # If label count doesn't match model output, avoid index errors.
        if hasattr(pred, "shape") and len(pred.shape) == 1:
            if len(self.labels) != int(pred.shape[0]):
                # Keep running, but clamp index to available labels.
                index = min(index, len(self.labels) - 1)
 
        return pred.tolist(), index

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300

# Initialize classifier (placeholder - replace with actual model)
try:
    classifier = SimpleClassifier(os.path.join("models", "keras_model.h5"), os.path.join("models", "labels.txt"))
except:
    print("Model files not found. Using placeholder classifier.")
    classifier = SimpleClassifier("", "")

labels = classifier.labels

prediction_interval_s = 0.2
last_prediction_time = 0.0
recent_indices = deque(maxlen=7)
last_shown_index = None
last_shown_conf = 0.0

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
            prediction = []
            index = -1
            confidence = 0.0
 
            now = time.time()
            if now - last_prediction_time >= prediction_interval_s:
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                last_prediction_time = now
 
                if prediction and index >= 0 and index < len(labels):
                    confidence = float(max(prediction))
                    recent_indices.append(index)
 
            if recent_indices:
                stable_index = Counter(recent_indices).most_common(1)[0][0]
                stable_conf = confidence if index == stable_index else last_shown_conf
                 
                if stable_index != last_shown_index or stable_conf != last_shown_conf:
                    print(f"Prediction: {labels[stable_index]}, Confidence: {stable_conf:.2f}")
                    last_shown_index = stable_index
                    last_shown_conf = stable_conf
            
            # Draw bounding box and label
            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0,255,0), cv2.FILLED)
            if last_shown_index is not None and last_shown_index < len(labels):
                cv2.putText(imgOutput, labels[last_shown_index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
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