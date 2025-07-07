import numpy as np
import cv2
import sys
import functools
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def disable_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect stdout to suppress print output
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            result = func(*args, **kwargs)
        # Restore stdout
        sys.stdout = sys.__stdout__
        return result
    return wrapper


class Detector:
    def __init__(self, *args, **kwargs):
        # No initialization needed for OpenCV detector
        print("Initialized lightweight OpenCV card detector")
        # Parameters for card detection
        self.min_card_area = 20000  # Minimum area for a card in pixels
        self.max_card_area = 500000  # Maximum area for a card in pixels
        self.min_aspect_ratio = 0.5  # Minimum aspect ratio (width/height)
        self.max_aspect_ratio = 1.5  # Maximum aspect ratio

    def detect_objects(self, img_path, scoreThreshold=0.5):
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return []
            
        # Get image dimensions
        img_h, img_w = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find cards
        detections = []
        for i, contour in enumerate(contours):
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_card_area or area > self.max_card_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = w / float(h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
                
            # Create mask for this contour
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Try to improve the contour with edge detection inside the bounding box
            roi = img[y:y+h, x:x+w]
            if roi.size == 0:  # Skip if ROI is empty
                continue
                
            # Add detection with confidence score based on contour quality
            # Calculate a confidence score based on how rectangular the contour is
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rect_area = cv2.contourArea(box)
            if rect_area == 0:  # Avoid division by zero
                continue
            confidence = min(area / rect_area, 1.0)  # How well the contour fits a rectangle
            
            # Only include if confidence is above threshold
            if confidence > scoreThreshold:
                detection = {
                    'bbox': [x, y, x + w, y + h],  # [x1, y1, x2, y2] format
                    'score': float(confidence),
                    'mask': mask
                }
                detections.append(detection)
        
        # If no cards were detected, return the whole image as a single card
        if not detections:
            mask = np.ones((img_h, img_w), dtype=np.uint8)
            detection = {
                'bbox': [0, 0, img_w, img_h],
                'score': 1.0,
                'mask': mask
            }
            detections.append(detection)
            
        return detections
