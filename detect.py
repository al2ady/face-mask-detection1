import argparse
import os
from mask_detection import MaskDetector
import cv2

def main():
    parser = argparse.ArgumentParser(description='Face Mask Detection')
    parser.add_argument('--source', type=str, required=True, choices=['webcam', 'image', 'video'],
                        help='Detection source: webcam, image, or video')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or video file')
    parser.add_argument('--model', type=str, default='models/mask_detector_model.h5',
                        help='Path to trained model (default: models/mask_detector_model.h5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output (for image/video processing)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please train the model first using train.py")
        return
    
    print("="*50)
    print("Face Mask Detection System")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print("="*50)
    
    # Initialize detector
    detector = MaskDetector(model_path=args.model)
    
    if args.source == 'webcam':
        print("\nStarting webcam detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        detector.run_webcam()
        
    elif args.source == 'image':
        if not args.input:
            print("Error: --input required for image processing")
            return
        
        print(f"\nProcessing image: {args.input}")
        results = detector.process_image(args.input)
        
        print("\nDetection Results:")
        for i, result in enumerate(results, 1):
            print(f"  Face {i}: {result['label']} (Confidence: {result['confidence']:.2%})")
        
        if args.output:
            # Save processed image
            frame = cv2.imread(args.input)
            frame = detector.draw_predictions(frame, results)
            cv2.imwrite(args.output, frame)
            print(f"\nOutput saved to: {args.output}")
    
    elif args.source == 'video':
        if not args.input:
            print("Error: --input required for video processing")
            return
        
        print(f"\nProcessing video: {args.input}")
        process_video(detector, args.input, args.output)

def process_video(detector, input_path, output_path=None):
    """Process video file"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
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
    print(f"Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and draw
        results = detector.detect_and_predict(frame)
        frame = detector.draw_predictions(frame, results)
        
        # Write frame if output specified
        if writer:
            writer.write(frame)
        
        # Display progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Show frame
        cv2.imshow('Video Processing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer:
        writer.release()
        print(f"\nOutput saved to: {output_path}")
    cv2.destroyAllWindows()
    
    print("\nVideo processing complete!")

if __name__ == "__main__":
    main()