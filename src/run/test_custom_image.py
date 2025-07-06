import sys
import os
import cv2
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import detector, scanner, viewer

def main():
    parser = argparse.ArgumentParser(description='Test card scanner with a custom image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--show', action='store_true', help='Show visualization of detection steps')
    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} does not exist")
        return

    # Initialize the detector
    card_detector = detector.Detector()
    
    # Read the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image {args.image}")
        return
    
    # Resize image if it's too large (optional)
    max_dimension = 1200
    height, width = image.shape[:2]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        print(f"Resized image to {image.shape[1]}x{image.shape[0]}")
    
    # Run detection
    print("Running card detection...")
    detections = card_detector.detect(image)
    print(f"Found {len(detections)} cards")
    
    # Process masks to extract card images
    print("Processing detected cards...")
    scanner.process_masks_to_cards(image, detections, mirror=False)
    
    # Generate hashes for detected cards
    scanner.hash_cards(detections)
    
    # Match hashes against the database
    scanner.match_hashes(detections)
    
    # Create a copy of the image for visualization
    result_image = image.copy()
    
    # Draw bounding boxes and labels
    scanner.draw_boxes(result_image, detections)
    scanner.write_card_labels(result_image, detections)
    
    # Display results
    print("\nResults:")
    for i, detection in enumerate(detections):
        if 'match' in detection:
            print(f"Card {i+1}: {detection['match']}")
        else:
            print(f"Card {i+1}: No match found")
    
    # Show the image with detections
    if args.show:
        cv2.imshow('Card Detection Results', result_image)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save the result image
    output_path = os.path.splitext(args.image)[0] + "_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved to {output_path}")

if __name__ == "__main__":
    main()
