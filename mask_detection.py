"""
Face Mask Detection System
Author: Your Name
Description: AI-powered real-time face mask detection using deep learning and computer vision
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

# ============================================
# PART 1: MODEL TRAINING
# ============================================

class MaskDetectionTrainer:
    """
    Trainer class for face mask detection model
    
    Args:
        data_dir: Path to dataset directory with train/ and validation/ folders
        img_size: Input image size (default: 224x224)
    """
    
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.model = None
        
    def create_model(self):
        """Create model using MobileNetV2 as base with transfer learning"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')  # 2 classes: with_mask, without_mask
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Model created successfully!")
        return model
    
    def prepare_data(self, batch_size=32):
        """
        Prepare training and validation data generators
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            train_generator, val_generator
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'validation'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train(self, epochs=10, batch_size=32):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.create_model()
        
        # Prepare data
        train_gen, val_gen = self.prepare_data(batch_size)
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print("\n🚀 Starting training...")
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        return history
    
    def save_model(self, path='mask_detector_model.h5'):
        """Save the trained model"""
        if self.model:
            self.model.save(path)
            print(f"💾 Model saved to {path}")
        else:
            print("❌ No model to save. Train the model first.")


# ============================================
# PART 2: REAL-TIME DETECTION
# ============================================

class MaskDetector:
    """
    Real-time face mask detector
    
    Args:
        model_path: Path to trained model file (.h5)
    """
    
    def __init__(self, model_path='mask_detector_model.h5'):
        # Load trained model
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load face detection classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Class labels and colors
        self.classes = ['with_mask', 'without_mask']
        self.colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255)    # Red
        }
    
    def detect_and_predict(self, frame):
        """
        Detect faces and predict mask status
        
        Args:
            frame: Input image/frame
            
        Returns:
            List of detection results with bbox, label, and confidence
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        results = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess for model
            face_resized = cv2.resize(face_roi, (224, 224))
            face_array = np.expand_dims(face_resized / 255.0, axis=0)
            
            # Predict
            predictions = self.model.predict(face_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            label = self.classes[class_idx]
            
            # Store result
            results.append({
                'bbox': (x, y, w, h),
                'label': label,
                'confidence': confidence
            })
        
        return results
    
    def draw_predictions(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input image/frame
            results: Detection results from detect_and_predict()
            
        Returns:
            Frame with annotations
        """
        for result in results:
            x, y, w, h = result['bbox']
            label = result['label']
            confidence = result['confidence']
            color = self.colors[label]
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label text
            text = f"{label}: {confidence*100:.1f}%"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, 
                (x, y - text_height - 10), 
                (x + text_width, y), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return frame
    
    def run_webcam(self):
        """Run real-time detection on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam")
            return
        
        print("\n📹 Webcam detection started!")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Detect and predict
            results = self.detect_and_predict(frame)
            
            # Draw results
            frame = self.draw_predictions(frame, results)
            
            # Add instructions
            cv2.putText(
                frame, "Press 'q' to quit | 's' to save", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Display frame
            cv2.imshow('Face Mask Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam detection stopped")
    
    def process_image(self, image_path, show=True, save_path=None):
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            show: Whether to display result
            save_path: Path to save output image (optional)
            
        Returns:
            Detection results
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Error: Cannot read image from {image_path}")
            return None
        
        # Detect and predict
        results = self.detect_and_predict(frame)
        
        # Draw results
        frame = self.draw_predictions(frame, results)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, frame)
            print(f"💾 Output saved to {save_path}")
        
        # Display if requested
        if show:
            cv2.imshow('Result', frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def process_video(self, video_path, output_path=None, show=True):
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show: Whether to display result while processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"\n🎬 Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect and draw
            results = self.detect_and_predict(frame)
            frame = self.draw_predictions(frame, results)
            
            # Write frame if output specified
            if writer:
                writer.write(frame)
            
            # Display progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Show frame if requested
            if show:
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ Processing interrupted by user")
                    break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"💾 Output video saved to: {output_path}")
        if show:
            cv2.destroyAllWindows()
        
        print("✅ Video processing complete!")


# ============================================
# USAGE EXAMPLES
# ============================================

def example_training():
    """Example: Train a new model"""
    print("="*60)
    print("TRAINING EXAMPLE")
    print("="*60)
    
    # Initialize trainer
    trainer = MaskDetectionTrainer(data_dir='dataset')
    
    # Create and train model
    trainer.create_model()
    history = trainer.train(epochs=15, batch_size=32)
    
    # Save model
    trainer.save_model('models/mask_detector_model.h5')
    
    print("\n✅ Training complete!")


def example_webcam_detection():
    """Example: Real-time webcam detection"""
    print("="*60)
    print("WEBCAM DETECTION EXAMPLE")
    print("="*60)
    
    # Initialize detector
    detector = MaskDetector(model_path='models/mask_detector_model.h5')
    
    # Run webcam detection
    detector.run_webcam()


def example_image_detection():
    """Example: Process single image"""
    print("="*60)
    print("IMAGE DETECTION EXAMPLE")
    print("="*60)
    
    # Initialize detector
    detector = MaskDetector(model_path='models/mask_detector_model.h5')
    
    # Process image
    results = detector.process_image(
        image_path='test_image.jpg',
        save_path='output_image.jpg'
    )
    
    # Print results
    print("\nDetection Results:")
    for i, result in enumerate(results, 1):
        print(f"  Face {i}: {result['label']} ({result['confidence']:.2%})")


def example_video_detection():
    """Example: Process video file"""
    print("="*60)
    print("VIDEO DETECTION EXAMPLE")
    print("="*60)
    
    # Initialize detector
    detector = MaskDetector(model_path='models/mask_detector_model.h5')
    
    # Process video
    detector.process_video(
        video_path='test_video.mp4',
        output_path='output_video.mp4',
        show=True
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACE MASK DETECTION SYSTEM")
    print("="*60)
    print("\nAvailable functions:")
    print("1. example_training() - Train a new model")
    print("2. example_webcam_detection() - Real-time webcam detection")
    print("3. example_image_detection() - Process single image")
    print("4. example_video_detection() - Process video file")
    print("\nUncomment the desired example below to run it:")
    print("="*60 + "\n")
    
    # Uncomment the example you want to run:
    # example_training()
    # example_webcam_detection()
    # example_image_detection()
    # example_video_detection()