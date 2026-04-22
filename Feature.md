# Roadmap: Human Behavior Detection Feature

## 1. Overview
This upcoming feature extends the project's capabilities from hand gesture recognition to **Human Behavior Analysis**. It will utilize facial landmark detection and upper-body posture analysis to classify the emotional and physical state of a person in real-time.

## 2. Target Labels
The model will be trained to recognize the following four primary behaviors:
- 😆 **Laugh**: Detecting wide mouth opening and eye "crinkling."
- 😡 **Angry**: Detecting furrowed brows and specific mouth tension.
- 😴 **Sleep**: Detecting prolonged eye closure (Drowsiness detection).
- 😐 **Normal**: The baseline/neutral state.

## 3. Proposed Directory Structure
To keep the project clean, this feature will reside in its own dedicated directory:

```text
behavior_detection/
├── models/
│   ├── behavior_model.h5      # CNN/RNN model for behavior classification
│   └── labels.txt             # [laugh, angry, sleep, normal]
├── src/
│   ├── preprocess.py          # Facial landmark extraction logic
│   └── utils.py               # Behavior-specific helper functions
├── collect_behavior_data.py   # Specialized script for facial data capture
└── behavior_inference.py      # Real-time behavior detection script (main)
```

## 4. Technical Strategy
### 4.1 Facial Landmark Integration
While the current project uses skin-color segmentation for hands, behavior detection will likely shift to **MediaPipe Face Mesh** or **Dlib**. This provides 468+ 3D landmarks for precise tracking of:
- **Eye Aspect Ratio (EAR)**: Specifically for the `sleep` label.
- **Mouth Opening Ratio (MOR)**: Specifically for the `laugh` label.
- **Brow Position**: Specifically for the `angry` label.

### 4.2 Data Collection Plan
- **Capture**: Similar to `data_collection.py`, but focused on the face.
- **Augmentation**: Applying random brightness and rotation to handle different environments.
- **Sequence Processing**: Potential use of a 3-5 frame window to differentiate between a "blink" and "sleeping."

## 5. Integration with Existing Project
The system can eventually run both models in parallel:
- **Left Window**: Sign Language Recognition.
- **Right Window**: Behavior Detection (User Sentiment).
- **Console**: Combined logs (e.g., "User is *Angry* while signing *No*").

---
*Status: Planned / In Design Phase*
