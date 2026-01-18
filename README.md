# Social Behavior Sign Language Recognition System

## Overview
A complete computer vision system for recognizing hand gestures using TensorFlow and OpenCV.

## Features
- Real-time hand detection using skin color segmentation
- Automated data collection with quality control
- CNN-based gesture classification
- Modular and configurable architecture

## Project Structure
```
d:\Works_Projects\
├── config.py                 # Configuration settings
├── hand_detector.py          # Hand detection module
├── enhanced_data_collection.py # Data collection script
├── train_model.py           # Model training script
├── gesture_classifier.py    # Real-time inference
├── models/                  # Trained models
│   ├── keras_model.h5      # TensorFlow model
│   └── labels.txt          # Gesture labels
├── Sign_data/              # Dataset directory
├── data_collection.py      # Original simple data collection
├── test_cvzone.py         # Original test script with cvzone
├── test.py                # Alternative test script
└── requirements.txt        # Dependencies
```

## Installation
```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Project

### Step 1: Data Collection (First Time)
```bash
python enhanced_data_collection.py
```

**What happens:**
- Creates `Sign_data/` folder automatically
- Collects 200 high-quality images per gesture
- Shows live camera feed with hand detection
- Only saves images when hand is properly detected
- Press 'q' to stop anytime

**Expected Output:**
```
Starting Hand Gesture Data Collection
Collecting 200 images per gesture
Gestures: Hello, I Love You, No, Please, Thank You, Yes

=== Collecting 'Hello' gestures ===
Position your hand in the camera frame
  Saved 1/200 images
  Saved 2/200 images
...
```

### Step 2: Train the Model
```bash
python train_model.py
```

**What happens:**
- Loads all collected gesture images
- Creates and trains a CNN model
- Shows training progress and accuracy
- Saves trained model to `models/keras_model.h5`
- Saves gesture labels to `models/labels.txt`

**Expected Output:**
```
Loading dataset...
  Loading Hello...
  Loading I Love You...
Loaded 1200 images
Training set: 960 images
Test set: 240 images
Model: "sequential"
_________________________________________________________________
Epoch 1/20
30/30 [==============================] - 15s 450ms/step - loss: 1.2345 - accuracy: 0.4567
...
Test accuracy: 0.8542
Model saved to models/keras_model.h5
```

### Step 3: Run Real-time Recognition
```bash
python gesture_classifier.py
```

**What happens:**
- Starts live camera for real-time gesture recognition
- Detects hands and predicts gestures instantly
- Shows confidence scores for each prediction
- Displays processed hand images
- Press 'q' to quit

**What you'll see:**
- Green bounding box around detected hand
- Gesture label with confidence percentage
- Separate window showing processed hand image

## 🎯 Quick Start Commands

```bash
# Complete workflow (run in order)
python enhanced_data_collection.py  # Collect training data
python train_model.py              # Train the model  
python gesture_classifier.py        # Test live recognition
```

## ⚡ Alternative: Test with Existing Model

If you want to test immediately without collecting new data:

```bash
# Use the existing test script with placeholder predictions
python test_cvzone.py
```

This uses your existing model structure with mock predictions for testing.

## 🔧 Configuration

Edit `config.py` to customize:
- Camera index (0 or 1)
- Dataset size (IMAGES_PER_GESTURE)
- Image resolution (IMAGE_SIZE)
- Hand detection sensitivity
- Gesture labels

## Supported Gestures
- Hello
- I Love You  
- No
- Please
- Thank You
- Yes

## Performance Tips

### For Better Data Collection:
1. **Lighting**: Use bright, consistent lighting
2. **Background**: Plain, non-skin-colored backgrounds work best
3. **Distance**: Keep hand at consistent distance from camera
4. **Variety**: Collect images from different angles and lighting

### For Better Recognition:
1. **Dataset Size**: More images = better accuracy
2. **Lighting**: Similar lighting to training conditions
3. **Hand Position**: Center hand in camera frame
4. **Background**: Consistent background helps detection

## Troubleshooting

### Camera Issues
```bash
# If camera doesn't work, try different indices
# Edit config.py and change CAMERA_INDEX from 0 to 1
```

### Model Loading Errors
```bash
# Ensure model files exist:
dir models/
# Should show: keras_model.h5 and labels.txt
```

### No Hand Detection
- Check lighting conditions
- Ensure hand is clearly visible in camera
- Try adjusting skin color ranges in `config.py`
- Make sure background doesn't contain skin-colored objects

### Low Accuracy
- Increase dataset size (collect more images)
- Add variety to training data (different angles, lighting)
- Ensure consistent hand positioning
- Check for mislabeled images in dataset

## File Descriptions

### Core Files:
- **`enhanced_data_collection.py`**: Professional data collection with quality control
- **`train_model.py`**: Complete CNN training pipeline
- **`gesture_classifier.py`**: Production-ready real-time recognition
- **`hand_detector.py`**: Unified hand detection module
- **`config.py`**: Centralized configuration settings

### Legacy Files:
- **`data_collection.py`**: Original simple data collection
- **`test_cvzone.py`**: Original test script with cvzone support
- **`test.py`**: Alternative test script using simple detection

## System Requirements
- Python 3.7+
- OpenCV 4.x
- TensorFlow 2.x
- Webcam or camera
- 4GB+ RAM recommended

## Future Enhancements
- [ ] Data augmentation pipeline for better training
- [ ] Multiple hand detection support
- [ ] Gesture sequence recognition
- [ ] Web interface for easier data collection
- [ ] Mobile deployment support
- [ ] Real-time performance optimization
- [ ] Additional gesture support
