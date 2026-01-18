import cv2
import numpy as np
import tensorflow as tf
from config import Config
from hand_detector import HandDetector

class GestureClassifier:
    def __init__(self):
        self.detector = HandDetector()
        self.model = None
        self.labels = Config.GESTURES
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(Config.MODEL_PATH)
            print(f"Model loaded successfully: {Config.MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model file exists and is compatible")
    
    def predict_gesture(self, image):
        """Predict gesture from processed hand image"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Preprocess image
            img_processed = image.astype(np.float32) / 255.0
            img_processed = np.expand_dims(img_processed, axis=0)
            
            # Predict
            predictions = self.model.predict(img_processed, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return self.labels[predicted_index], confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def run_inference(self):
        """Run real-time gesture recognition"""
        print("Starting Gesture Recognition")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            img_output = frame.copy()
            
            # Detect hand
            hand_detected, contour = self.detector.detect_hand(frame)
            
            if hand_detected:
                # Process hand
                processed_hand = self.detector.crop_and_resize(frame, contour)
                
                if processed_hand is not None:
                    # Predict gesture
                    gesture, confidence = self.predict_gesture(processed_hand)
                    
                    if gesture:
                        # Draw results
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Bounding box
                        cv2.rectangle(img_output, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
                        
                        # Background for text
                        cv2.rectangle(img_output, (x-20, y-70), (x+380, y-30), (0, 255, 0), -1)
                        
                        # Gesture label and confidence
                        text = f"{gesture} ({confidence:.2f})"
                        cv2.putText(img_output, text, (x, y-40), 
                                   cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
                        
                        # Show processed hand
                        cv2.imshow('Processed Hand', processed_hand)
            else:
                cv2.putText(img_output, "No Hand Detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Gesture Recognition', img_output)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    classifier = GestureClassifier()
    classifier.run_inference()
