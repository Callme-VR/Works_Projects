import os
import cv2
import time
import uuid
import numpy as np
from config import Config
from hand_detector import HandDetector

class DataCollector:
    def __init__(self):
        self.detector = HandDetector()
        self.cap = None
        
    def start_collection(self):
        """Main data collection process"""
        print("Starting Hand Gesture Data Collection")
        print(f"Collecting {Config.IMAGES_PER_GESTURE} images per gesture")
        print(f"Gestures: {', '.join(Config.GESTURES)}")
        
        # Create dataset directory
        os.makedirs(Config.DATASET_PATH, exist_ok=True)
        
        for gesture in Config.GESTURES:
            self._collect_gesture_images(gesture)
        
        print("Data collection completed!")
    
    def _collect_gesture_images(self, gesture):
        """Collect images for a specific gesture"""
        gesture_dir = os.path.join(Config.DATASET_PATH, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        print(f"\n=== Collecting '{gesture}' gestures ===")
        print("Position your hand in the camera frame")
        time.sleep(3)
        
        if not self.cap:
            self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        collected = 0
        last_capture = 0
        
        while collected < Config.IMAGES_PER_GESTURE:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Detect hand
            hand_detected, contour = self.detector.detect_hand(frame)
            
            if hand_detected:
                # Process hand image
                processed_hand = self.detector.crop_and_resize(frame, contour)
                
                if processed_hand is not None:
                    # Draw bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
                    cv2.putText(frame, f"Capturing... {collected+1}/{Config.IMAGES_PER_GESTURE}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Capture with delay
                    current_time = time.time()
                    if current_time - last_capture > Config.CAPTURE_DELAY:
                        # Save image
                        filename = f"{gesture}_{uuid.uuid4()}.jpg"
                        filepath = os.path.join(gesture_dir, filename)
                        cv2.imwrite(filepath, processed_hand)
                        
                        collected += 1
                        last_capture = current_time
                        print(f"  Saved {collected}/{Config.IMAGES_PER_GESTURE} images")
                        
                        # Show processed image
                        cv2.imshow('Processed Hand', processed_hand)
            else:
                cv2.putText(frame, "No Hand Detected - Show your hand", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Live Feed', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Stopping collection for '{gesture}'")
                break
        
        cv2.destroyAllWindows()
        print(f"Completed '{gesture}': {collected} images collected")

if __name__ == "__main__":
    collector = DataCollector()
    collector.start_collection()
