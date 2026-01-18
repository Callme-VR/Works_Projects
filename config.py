# Hand Gesture Recognition System - Configuration

class Config:
    # Camera settings
    CAMERA_INDEX = 0
    IMAGE_SIZE = 300
    OFFSET = 20
    
    # Data collection
    DATASET_PATH = "Sign_data"
    IMAGES_PER_GESTURE = 200
    CAPTURE_DELAY = 1.0
    
    # Model settings
    MODEL_PATH = "models/keras_model.h5"
    LABELS_PATH = "models/labels.txt"
    
    # Gesture labels (consistent with model)
    GESTURES = ["Hello", "I Love You", "No", "Please", "Thank You", "Yes"]
    
    # Hand detection
    MIN_CONTOUR_AREA = 1000
    SKIN_COLOR_RANGE = {
        'lower': [0, 20, 70],
        'upper': [20, 255, 255]
    }
