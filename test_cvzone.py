import cv2
import numpy as np
import math
import time
import os

# Try cvzone first, fallback to simple detection if it fails
try:
    from cvzone.HandTrackingModule import HandDetector
    from cvzone.ClassificationModule import Classifier
    CVZONE_AVAILABLE = True
    print("Using cvzone for hand detection")
except ImportError:
    print("cvzone not available, using simple hand detection")
    CVZONE_AVAILABLE = False

# Simple hand detection fallback (same as data collection)
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

# Simple placeholder classifier
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
counter = 0

labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]

# Initialize detector and classifier
if CVZONE_AVAILABLE:
    try:
        detector = HandDetector(maxHands=1)
        classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        print("cvzone initialized successfully")
    except Exception as e:
        print(f"cvzone initialization failed: {e}")
        CVZONE_AVAILABLE = False

if not CVZONE_AVAILABLE:
    classifier = SimpleClassifier("Model/keras_model.h5", "Model/labels.txt")

while True:
    success, img = cap.read()
    if not success:
        continue
        
    imgOutput = img.copy()
    
    if CVZONE_AVAILABLE:
        # Use cvzone hand detection
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            
            # Ensure crop bounds are within image
            y_start = max(0, y-offset)
            y_end = min(img.shape[0], y+h+offset)
            x_start = max(0, x-offset)
            x_end = min(img.shape[1], x+w+offset)
            
            imgCrop = img[y_start:y_end, x_start:x_end]
            
            if imgCrop.size > 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(f"Prediction: {labels[index]}, Confidence: {max(prediction):.2f}")

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(f"Prediction: {labels[index]}, Confidence: {max(prediction):.2f}")

                # Draw bounding box and label
                cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0,255,0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
                cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)
    else:
        # Use simple hand detection (fallback)
        hand_detected, contour = detect_hand(img)
        
        if hand_detected:
            x, y, w, h = cv2.boundingRect(contour)
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            
            # Ensure crop bounds are within image
            y_start = max(0, y-offset)
            y_end = min(img.shape[0], y+h+offset)
            x_start = max(0, x-offset)
            x_end = min(img.shape[1], x+w+offset)
            
            imgCrop = img[y_start:y_end, x_start:x_end]
            
            if imgCrop.size > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(f"Prediction: {labels[index]}, Confidence: {max(prediction):.2f}")

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(f"Prediction: {labels[index]}, Confidence: {max(prediction):.2f}")

                # Draw bounding box and label
                cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0,255,0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
                cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)
        else:
            cv2.putText(imgOutput, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Image', imgOutput)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
