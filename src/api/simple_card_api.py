import os
import sys
import json
import base64
import tempfile
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import imagehash
from flask import Flask, request, jsonify

# Add the parent directory to the path so we can import from tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import detector, scanner

# Configuration
hash_size = 16  # bytes - must match the scanner.py setting
temp_dir = tempfile.gettempdir()  # Temporary directory for saving images

# Support both Docker and local development paths
if os.path.exists("/app/data/hashes_dphash_16.json"):
    hash_db_file = "/app/data/hashes_dphash_16.json"  # Docker path
else:
    hash_db_file = "../data/hashes_dphash_16.json"  # Local development path

# Similarly for API key file
if os.path.exists("/app/credentials.json"):
    api_key_file = "/app/credentials.json"  # Docker path
else:
    api_key_file = "../../credentials.json"  # Local development path

# Initialize the detector
print("Initializing card detector...")
try:
    # Get device from environment variable or default to CPU
    import torch
    device = os.environ.get('PYTORCH_DEVICE', 'cpu')
    print(f"Using device: {device}")
    
    # Set PyTorch device globally
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(':')[1]) if ':' in device else 0)
    
    # Use MMDetection model for card detection
    detector_model = "rtmdet-ins_tiny_8xb32-300e_coco"
    detector_weights = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20220902_112414-78d0f50f.pth"
    
    # Initialize the detector
    # The detector will use the global PyTorch device setting
    card_detector = detector.Detector(detector_model, detector_weights)
    
    # Function to detect cards using the detector
    def detect_cards(image_path):
        return card_detector.detect_objects(image_path, scoreThreshold=0.5)
    
    print("Card detector initialized successfully!")
except Exception as e:
    print(f"Warning: Could not initialize card detector: {e}")
    print("Falling back to single card detection mode")
    card_detector = None
    
    # Fallback function that returns the whole image as a single card
    def detect_cards(image_path):
        # Return the whole image as a single card
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        return [{
            'bbox': [0, 0, w, h],
            'score': 1.0,
            'mask': np.ones((h, w), dtype=np.uint8)
        }]

# Initialize Flask app
app = Flask(__name__)

# Load hash database
def load_hash_db():
    if os.path.exists(hash_db_file):
        with open(hash_db_file, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    else:
        print(f"Error: Hash database {hash_db_file} not found")
        return {}

# Load PokemonTCG API key
def load_api_key():
    try:
        # First try to get API key from environment variable
        api_key = os.environ.get('POKEMON_TCG_API_KEY')
        if api_key:
            return api_key
            
        # Fall back to credentials file if environment variable is not set
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r') as f:
                credentials = json.load(f)
                return credentials.get('api_key')
                
        return None
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

# Get card market data from PokemonTCG API
def get_card_market_data(card_id):
    try:
        from pokemontcgsdk import Card
        from pokemontcgsdk import RestClient
        
        api_key = load_api_key()
        if not api_key:
            return {"error": "No API key found"}
        
        RestClient.configure(api_key)
        
        # Parse set code and card number from card_id
        parts = card_id.split('-')
        if len(parts) != 2:
            return {"error": "Invalid card ID format"}
        
        set_code, card_number = parts
        
        # Query the API for the card
        query = f'set.id:{set_code} number:{card_number}'
        cards = Card.where(q=query)
        
        if not cards or len(cards) == 0:
            return {"error": f"No card found for {card_id}"}
        
        card = cards[0]
        
        # Extract market data
        market_data = {}
        if hasattr(card, 'tcgplayer') and hasattr(card.tcgplayer, 'prices'):
            market_data['tcgplayer'] = card.tcgplayer.prices
        
        if hasattr(card, 'cardmarket') and hasattr(card.cardmarket, 'prices'):
            market_data['cardmarket'] = card.cardmarket.prices
        
        return {
            "id": card_id,
            "name": card.name,
            "set": card.set.name,
            "rarity": getattr(card, 'rarity', 'Unknown'),
            "market_data": market_data
        }
    except Exception as e:
        return {"error": f"Error fetching market data: {str(e)}"}

# Generate hash for an image
def hash_image(img):
    img = img.convert('RGB')
    dhash = imagehash.dhash(img, hash_size=hash_size)
    phash = imagehash.phash(img, hash_size=hash_size)
    hash_str = f'{dhash}{phash}'
    return hash_str

# Calculate Hamming distance between two hashes
def hamming_distance(hash1, hash2):
    # Convert hex strings to binary
    bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
    
    # Count differences
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

# Find best match for a hash in the database
def find_match(card_hash, hash_dict, threshold=10):
    best_match = None
    best_distance = float('inf')
    
    for card_id, card_data in hash_dict.items():
        if 'hash' in card_data:
            distance = hamming_distance(card_hash, card_data['hash'])
            if distance < best_distance:
                best_distance = distance
                best_match = card_data
                best_match['id'] = card_id
    
    if best_match and best_distance <= threshold:
        return best_match, best_distance
    return None, best_distance

# Process image and identify cards
def process_image(image_data):
    try:
        # Check if the image_data is properly formatted
        if ',' in image_data:
            # Handle data URLs like "data:image/jpeg;base64,/9j/4AAQSkZ..."
            image_data = image_data.split(',', 1)[1]
        
        # Strip any whitespace
        image_data = image_data.strip()
        
        # Convert base64 to image
        try:
            img_bytes = base64.b64decode(image_data)
            img_io = BytesIO(img_bytes)
            pil_img = Image.open(img_io)
            
            # Convert PIL Image to OpenCV format
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as img_error:
            return {
                "success": False,
                "error": f"Invalid base64 image data: {str(img_error)}",
                "help": "Make sure you're sending a valid base64-encoded image without any wrapping or formatting issues."
            }
        
        # Save temp image for detector
        temp_img_path = os.path.join(temp_dir, "temp_card_image.jpg")
        cv2.imwrite(temp_img_path, img)
        
        # Detect cards in the image
        detections = detect_cards(temp_img_path)
        
        if not detections or len(detections) == 0:
            return {
                "success": False,
                "error": "No cards detected in the image"
            }
        
        # Load hash database
        hash_dict = load_hash_db()
        if not hash_dict:
            return {
                "success": False,
                "error": "Hash database is empty or could not be loaded",
                "path_checked": hash_db_file
            }
        
        # Process each detected card
        results = []
        for i, detection in enumerate(detections):
            # Extract card from image
            card_img = extract_card(img, detection)
            
            if card_img is None:
                results.append({
                    "success": False,
                    "card_index": i,
                    "error": "Failed to extract card from image",
                    "bbox": detection['bbox']
                })
                continue
            
            # Convert OpenCV image to PIL for hashing
            pil_card = Image.fromarray(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
            
            # Generate hash
            card_hash = hash_image(pil_card)
            
            # Find match
            match, distance = find_match(card_hash, hash_dict)
            
            if match:
                # Get market data
                market_data = get_card_market_data(match['id'])
                
                results.append({
                    "success": True,
                    "card_index": i,
                    "card_id": match['id'],
                    "name": match['name'],
                    "match_confidence": (100 - (distance * 5)) if distance <= 10 else 0,
                    "market_data": market_data,
                    "bbox": detection['bbox']
                })
            else:
                results.append({
                    "success": False,
                    "card_index": i,
                    "error": "No match found for this card",
                    "best_distance": distance,
                    "bbox": detection['bbox']
                })
        
        # Clean up
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        return {
            "success": True,
            "cards": results,
            "card_count": len(results)
        }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Error processing image: {str(e)}",
            "traceback": traceback.format_exc()
        }

# Extract card from image using detection information
def extract_card(img, detection):
    try:
        # Get the bounding box
        x1, y1, x2, y2 = detection['bbox']
        
        # Crop the image to the bounding box
        card_img = img[y1:y2, x1:x2]
        
        # If we have a mask, apply it
        if 'mask' in detection and detection['mask'] is not None:
            # Resize mask to match the bounding box size
            mask = detection['mask']
            mask_roi = mask[y1:y2, x1:x2]
            
            # Apply the mask
            card_img = cv2.bitwise_and(card_img, card_img, mask=mask_roi)
        
        return card_img
    except Exception as e:
        print(f"Error extracting card: {e}")
        return None

# API endpoint for card identification
@app.route('/identify', methods=['POST'])
def identify_cards():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        # Process the image and get results
        result = process_image(data['image'])
        
        # Log the number of cards detected
        if 'card_count' in result:
            print(f"Detected {result['card_count']} cards in the image")
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": f"Error processing request: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

# API endpoint for card market data
@app.route('/market-data/<card_id>', methods=['GET'])
def get_market_data(card_id):
    try:
        result = get_card_market_data(card_id)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
