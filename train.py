import argparse
import os
from mask_detection import MaskDetectionTrainer
import matplotlib.pyplot as plt

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Face Mask Detection Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--output', type=str, default='models/mask_detector_model.h5',
                        help='Path to save trained model (default: models/mask_detector_model.h5)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for training (default: 224)')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("="*50)
    print("Face Mask Detection - Model Training")
    print("="*50)
    print(f"Dataset: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Output Model: {args.output}")
    print("="*50)
    
    # Initialize trainer
    trainer = MaskDetectionTrainer(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size)
    )
    
    # Create model
    print("\nCreating model...")
    model = trainer.create_model()
    model.summary()
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(args.output)
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(history, 'training_history.png')
    
    # Print final metrics
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()