import os
import sys
import json
import base64
import requests
from PIL import Image
import io

# Add the parent directory to the path so we can import from the API
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def test_multi_card_detection(image_path):
    """Test the multi-card detection API with a given image"""
    # Encode the image
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the request
    url = "http://localhost:5000/identify"
    headers = {"Content-Type": "application/json"}
    data = {"image": base64_image}
    
    # Send the request
    print(f"Sending request with image: {image_path}")
    response = requests.post(url, headers=headers, json=data)
    
    # Print the response
    print(f"Status code: {response.status_code}")
    result = response.json()
    
    # Pretty print the result
    print(json.dumps(result, indent=2))
    
    # Return the result for further processing if needed
    return result

if __name__ == "__main__":
    # Check if an image path was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to a test image if none provided
        print("No image path provided. Please provide a path to an image with multiple cards.")
        print("Usage: python test_multi_card.py <path_to_image>")
        sys.exit(1)
    
    # Test the API
    test_multi_card_detection(image_path)
