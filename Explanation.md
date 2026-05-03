# 📄 Project Code Explanation

This document provides a detailed, line-by-line explanation of the core scripts in the Sign Language and Human Behavior Recognition system.

---

## 🛠️ 1. `data_collection.py` (Sign Language Data)
This script is used to capture hand gesture images and preprocess them for training.

### Imports & Configuration
- **Line 1-6**: Imports standard libraries: `os` (file paths), `cv2` (OpenCV), `time` (delays), `uuid` (unique filenames), `numpy` (arrays), and `math` (calculations).
- **Line 9-10**: `image_path` defines where to save data; `labels` lists the gestures to collect.
- **Line 12-15**: Constants for collection: `number_of_images` per label, `offset` (padding), and `image_size` (300x300 target).

### Functions
- **`detect_hand(frame)`**:
  - Converts BGR to **HSV** color space for better skin segmentation.
  - Applies a mask using `lower_skin` and `upper_skin` ranges.
  - Finds contours and selects the **largest one** (assumed to be the hand) if it's over 1000 pixels in area.
- **`crop_and_resize_hand(frame, contour)`**:
  - Gets the bounding box (`x, y, w, h`) of the hand.
  - Crops the region with `offset` padding.
  - Creates a **300x300 white background**.
  - Calculates the **aspect ratio**:
    - If height > width: Resizes hand to height 300 and centers it horizontally.
    - If width > height: Resizes hand to width 300 and centers it vertically.

### Main Loop
- Iterates through each `label` in the list.
- **`cv2.VideoCapture(0)`**: Opens the default webcam.
- **Inner Loop**: Captures frames until `number_of_images` (15) is reached.
- **`cv2.imwrite`**: Saves the processed 300x300 image with a unique ID (`uuid`).
- **`cv2.waitKey(1)`**: Listens for 'q' to skip or quit.

---

## 👤 2. `data-collection2.py` (Behavior Data)
Identical logic to `data_collection.py` but specialized for **Face Detection**.

### Key Differences
- **Line 16**: Initializes `cv2.CascadeClassifier` using the **Haar Cascade** for frontal faces.
- **`detect_face(frame)`**: Uses `face_cascade.detectMultiScale` to find faces in grayscale. It picks the largest face found.
- **`crop_and_resize_face`**: Similar aspect-ratio-aware resizing as the hand script, but optimized for face crops.
- **Labels**: Focuses on `smile`, `angry`, `normal`, and `sad`.

---

## 🧠 3. `test.py` (Sign Language Inference)
This script performs real-time recognition using the trained model.

### `SimpleClassifier` Class
- **`__init__`**: Loads the Keras model (`keras_model.h5`) and `labels.txt`.
- **`getPrediction(img)`**:
  - Resizes the 300x300 image to the model's required input size.
  - Normalizes pixels to 0-1 (`/ 255.0`).
  - **`model.predict`**: Runs the image through the neural network.
  - Returns the **index** of the highest probability.

### Main Loop & Smoothing
- **`deque(maxlen=7)`**: Stores the last 7 predictions.
- **`Counter().most_common(1)`**: Implements **Temporal Smoothing**. It picks the most frequent label from the deque to prevent "flickering" predictions.
- **Visualization**:
  - Draws a green rectangle around the hand.
  - Displays the predicted label text above the hand.
  - Shows the `ImageWhite` (what the model sees) in a separate window.

---

## 🎭 4. `test2.py` (Behavior Inference)
Real-time emotion/behavior detection.

### Logic Flow
1.  **Detect**: Uses Haar Cascades to find the face.
2.  **Process**: Crops and resizes the face to 300x300 on a white background.
3.  **Predict**: Passes the image to the behavior model (`model/keras_model.h5`).
4.  **Display**: Overlays the emotion (e.g., "smile", "angry") on the live video feed.

---

## 🚀 Technical Highlights
- **Skin Segmentation**: Uses HSV instead of RGB to be more robust against lighting changes.
- **White Padding**: Ensures the hand/face is always centered and consistent, which significantly improves model accuracy.
- **Smoothing**: The use of a `deque` ensures that even if one frame is misclassified, the displayed result remains stable.
