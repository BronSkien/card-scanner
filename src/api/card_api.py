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
hash_db_file = "../data/hashes_dphash_16.json"  # Path to the hash database
api_key_file = "../../credentials.json"  # Path to PokemonTCG API key file
temp_dir = tempfile.gettempdir()  # Temporary directory for saving images

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

# Process image and identify cards
def process_image(image_data):
    try:
        # Convert base64 to image
        img_bytes = base64.b64decode(image_data)
        img_io = BytesIO(img_bytes)
        pil_img = Image.open(img_io)
        
        # Convert PIL Image to OpenCV format
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save temp image for detector
        temp_img_path = os.path.join(temp_dir, "temp_card_image.jpg")
        cv2.imwrite(temp_img_path, img)
        
        # Detect cards in the image
        detections = detector.detect_cards(temp_img_path)
        
        if not detections or len(detections) == 0:
            return {"error": "No cards detected in the image"}
        
        # Load hash database
        hash_dict = load_hash_db()
        
        # Process each detected card
        results = []
        for i, detection in enumerate(detections):
            # Extract card from image
            card_img = scanner.extract_card(img, detection)
            
            # Generate hash
            card_hash = scanner.hash_image(card_img)
            
            # Find match
            match, distance = scanner.find_match(card_hash, hash_dict)
            
            if match:
                # Get market data
                market_data = get_card_market_data(match['id'])
                
                results.append({
                    "card_index": i,
                    "card_id": match['id'],
                    "name": match['name'],
                    "match_confidence": (100 - (distance * 5)) if distance <= 10 else 0,
                    "market_data": market_data
                })
            else:
                results.append({
                    "card_index": i,
                    "error": "No match found for this card"
                })
        
        # Clean up
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        return {"cards": results}
    
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

# API endpoint for card identification
@app.route('/identify', methods=['POST'])
def identify_cards():
    try:
        # Check if request has the image
        if 'image' not in request.json:
            return jsonify({"error": "No image provided"}), 400
        
        # Get base64 encoded image
        image_data = request.json['image']
        
        # Process the image
        result = process_image(image_data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

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
