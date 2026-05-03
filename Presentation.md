# 🎓 Project Presentation: Multi-Modal Human Interaction System

**Title:** Real-time Sign Language Recognition and Human Behavior Analysis  
**Academic Year:** 2026-2027  
**Project Lead:** [Vishal Rajput]  
**Objective:** Bridging communication gaps using Computer Vision and Deep Learning for Deaf and Dump.

---

## 📽️ Slide-by-Slide Outline
### Slide 1: Title Slide
*   **Main Title:** Multi-Modal Human Interaction & Recognition System
*   **Sub-title:** A Software-Agnostic Approach to Non-Verbal Communication
*   **Details:** [Your Name], [Roll Number], Department of Computer Science.--

### Slide 2: Introduction
*   **Concept:** A system that interprets human intentions through gestures (Sign Language) and emotions (Facial Behavior).
*   **Why Multi-Modal?** Combining "What" is being said (Sign) with "How" it is felt (Behavior) provides a holistic understanding of human input.


### Slide 3: Problem Statement
*   **The Barrier:** 70 million+ deaf people worldwide face daily communication challenges.
*   **Contextual Blindness:** Most AI systems ignore emotional context behind a gesture.
*   **Cost Issue:** High-accuracy systems often requir
e expensive sensors (LIDAR). Our solution needs only a standard 2D webcam.

### Slide 4: Proposed Solution
*   **Digital Interpreter:** A real-time software pipeline that detects and classifies:
    *   **Gestures:** Hello, Yes, No, Thank You, I Love You, Please.
    *   **Behaviors:** Smile, Angry, Normal, Sad.
*   **Core Value:** Zero-hardware cost, low latency, and high portability (Dockerized).


### Slide 5: System Architecture
*   **Pipeline:** `Webcam Feed` → `Preprocessing (HSV/Haar)` → `ROI Extraction` → `Model Inference` → `Temporal Smoothing` → `Visual UI`.
*   **Modular Design:** Two distinct pipelines (SLR and HBA) that can run independently or in parallel.

### Slide 6: Methodology - Preprocessing
*   **Standardization:** All inputs are normalized to a 300x300 white canvas.
*   **Aspect Ratio Preservation:** We calculate scale factors and pad with a white background to prevent distortion—critical for CNN accuracy.
*   **Segmentation:** 
    *   **Hands:** HSV Skin-tone segmentation for robust tracking.
    *   **Face:** Haar Cascade Frontal Face detection for speed.

### Slide 7: Deep Learning Implementation
*   **Framework:** TensorFlow/Keras.
*   **Model Architecture:** Convolutional Neural Networks (CNNs).
*   **Temporal Smoothing:** Implementation of a **Deque** and **Majority Voting** algorithm. We average the last 7 frames to ensure a "zero-flicker" prediction display.

### Slide 8: Tools & Technologies
*   **Language:** Python 3.11
*   **Computer Vision:** OpenCV, MediaPipe
*   **Machine Learning:** TensorFlow, Keras
*   **Deployment:** Docker (Standardized environment)

### Slide 9: Results & Performance
*   **Accuracy:** Robust performance in controlled lighting.
*   **Latency:** Real-time inference (30+ FPS) on standard CPU-based laptops.
*   **Utility:** Automated data collection scripts allow for rapid training of new classes.

### Slide 10: Future Scope
*   **Healthcare:** Monitoring patient responsiveness in telemedicine.
*   **EdTech:** Interactive tools for learning Sign Language.
*   **Mobile:** Conversion to TFLite for Android/iOS deployment.

---

## 🎙️ The Pitch (Suggested Script)

>
> My system uses two specialized pipelines. First, the **Sign Language module** uses skin-tone segmentation to isolate the hand and predict gestures with high stability using a temporal smoothing algorithm. Second, the **Behavior module** uses Haar Cascades to analyze facial expressions in real-time.
>
> What makes this project unique is the **'Training-Inference Parity'**. I've developed custom data collection scripts that ensure the data we train on is identical in format to the data we see in real-time. This eliminates the 'real-world vs dataset' gap that many AI projects face."

---

## ❓ Viva / Evaluation Success Guide (Q&A)

**Q1: Why use HSV segmentation for hands instead of just RGB?**
*   **A:** RGB is highly sensitive to lighting. HSV (Hue, Saturation, Value) separates color (Hue) from intensity (Value), making skin-detection much more robust under different lighting conditions.

**Q2: How do you handle "flickering" in predictions?**
*   **A:** I implemented a **Majority Voting** system using a `deque` of size 7. The system stores the last 7 predictions and only updates the display label if a specific class appears most frequently in that window.

**Q3: Why 300x300 with a white background?**
*   **A:** Neural networks require fixed-size inputs. If we just resize a narrow hand crop to a square, the hand becomes distorted. Placing it on a 300x300 white canvas preserves the **original aspect ratio**, which is vital for the model to recognize finger positions accurately.

**Q4: What is the role of Docker here?**
*   **A:** Machine Learning projects often fail due to version conflicts. Docker ensures the project runs identically on your machine as it does on mine by containerizing the entire environment and its dependencies.

---

## 🛠️ Deployment Instructions (For Demo)

1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Sign Language Recognition:**
    ```bash
    python test.py
    ```
3.  **Run Behavior Analysis:**
    ```bash
    python test2.py
    ```
