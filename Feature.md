# Roadmap: Human Behavior Detection Feature

## 1. Overview
This feature extends the project's capabilities from hand gesture recognition to **Human Behavior Analysis**. It utilizes facial detection and expression analysis to classify the emotional state of a person in real-time.

## 2. Target Labels
The model is trained to recognize the following four primary behaviors:
- 😊 **Smile**: Detecting positive emotional state.
- 😡 **Angry**: Detecting furrowed brows and specific mouth tension.
- 😐 **Normal**: The baseline/neutral state.
- 😔 **Sad**: Detecting negative emotional state.

## 3. Directory Structure
The feature is implemented using the following components:
- `model/keras_model.h5`: CNN model for behavior classification.
- `model/labels.txt`: [smile, angry, normal, sad].
- `data-collection2.py`: Specialized script for facial data capture.
- `test2.py`: Real-time behavior detection script.

## 4. Technical Strategy
### 4.1 Facial Detection
The system uses **OpenCV Haar Cascades** (`haarcascade_frontalface_default.xml`) for robust and fast face detection. This allows the system to run on hardware with limited resources.

### 4.2 Data Collection
- **Capture**: `data-collection2.py` captures 15 images per label with a 1-second delay between captures.
- **Preprocessing**: Each face is cropped, padded to maintain aspect ratio, and resized to 300x300 pixels on a white background.

## 5. Integration
The system currently runs as a standalone module but is architecturally compatible with the Sign Language Recognition module. Future versions will merge both into a unified multi-modal dashboard.

---
*Status: Implemented*

