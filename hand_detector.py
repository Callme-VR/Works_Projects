import cv2
import numpy as np
import math
from config import Config

class HandDetector:
    def __init__(self):
        self.offset = Config.OFFSET
        self.image_size = Config.IMAGE_SIZE
        self.min_area = Config.MIN_CONTOUR_AREA
        
    def detect_hand(self, frame):
        """Detect hand using skin color detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color mask
        lower_skin = np.array(Config.SKIN_COLOR_RANGE['lower'], dtype=np.uint8)
        upper_skin = np.array(Config.SKIN_COLOR_RANGE['upper'], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_area:
                return True, largest_contour
        
        return False, None
    
    def crop_and_resize(self, frame, contour):
        """Crop and resize hand region to standard size"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ensure bounds within image
        img_h, img_w = frame.shape[:2]
        y_start = max(0, y - self.offset)
        y_end = min(img_h, y + h + self.offset)
        x_start = max(0, x - self.offset)
        x_end = min(img_w, x + w + self.offset)
        
        image_crop = frame[y_start:y_end, x_start:x_end]
        
        if image_crop.size == 0:
            return None
        
        # Create white background
        image_white = np.ones((self.image_size, self.image_size, 3), np.uint8) * 255
        
        # Resize maintaining aspect ratio
        aspect_ratio = h / w if w > 0 else 1
        
        if aspect_ratio > 1:
            k = self.image_size / h
            w_cal = math.ceil(k * w)
            image_resize = cv2.resize(image_crop, (w_cal, self.image_size))
            w_gap = math.ceil((self.image_size - w_cal) / 2)
            image_white[:, w_gap:w_gap + w_cal] = image_resize
        else:
            k = self.image_size / w
            h_cal = math.ceil(k * h)
            image_resize = cv2.resize(image_crop, (self.image_size, h_cal))
            h_gap = math.ceil((self.image_size - h_cal) / 2)
            image_white[h_gap:h_gap + h_cal, :] = image_resize
        
        return image_white
