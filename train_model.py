import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import Config

class GestureTrainer:
    def __init__(self):
        self.image_size = Config.IMAGE_SIZE
        self.gestures = Config.GESTURES
        self.dataset_path = Config.DATASET_PATH
        
    def load_dataset(self):
        """Load and preprocess the dataset"""
        images = []
        labels = []
        
        print("Loading dataset...")
        
        for gesture in self.gestures:
            gesture_dir = os.path.join(self.dataset_path, gesture)
            if not os.path.exists(gesture_dir):
                print(f"Warning: No data found for gesture '{gesture}'")
                continue
            
            print(f"  Loading {gesture}...")
            for filename in os.listdir(gesture_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(gesture_dir, filename)
                    
                    # Load and preprocess image
                    img = cv2.imread(filepath)
                    if img is not None:
                        img = cv2.resize(img, (self.image_size, self.image_size))
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(gesture)
        
        print(f"Loaded {len(images)} images")
        return np.array(images), np.array(labels)
    
    def create_model(self):
        """Create CNN model for gesture recognition"""
        model = tf.keras.Sequential([
            # Convolutional layers
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size, self.image_size, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.gestures), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train_model(self):
        """Train the gesture recognition model"""
        # Load data
        X, y = self.load_dataset()
        
        if len(X) == 0:
            print("No training data found!")
            return
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Create model
        model = self.create_model()
        model.summary()
        
        # Train
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save(Config.MODEL_PATH)
        
        # Save labels
        with open(Config.LABELS_PATH, 'w') as f:
            for i, label in enumerate(label_encoder.classes_):
                f.write(f"{i} {label}\n")
        
        print(f"Model saved to {Config.MODEL_PATH}")
        print(f"Labels saved to {Config.LABELS_PATH}")
        
        return model, history

if __name__ == "__main__":
    trainer = GestureTrainer()
    model, history = trainer.train_model()
