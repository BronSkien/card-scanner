import os
import sys
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import imagehash
from flask import Flask, request, jsonify

# Configuration
hash_size = 16  # bytes - must match the scanner.py setting
hash_db_file = "../data/hashes_dphash_16.json"  # Path to the hash database
api_key_file = "../../credentials.json"  # Path to PokemonTCG API key file

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
            img = Image.open(img_io)
        except Exception as img_error:
            return {
                "success": False,
                "error": f"Invalid base64 image data: {str(img_error)}",
                "help": "Make sure you're sending a valid base64-encoded image without any wrapping or formatting issues."
            }
        
        # Load hash database
        hash_dict = load_hash_db()
        if not hash_dict:
            return {
                "success": False,
                "error": "Hash database is empty or could not be loaded",
                "path_checked": hash_db_file
            }
        
        # Generate hash
        card_hash = hash_image(img)
        
        # Find match
        match, distance = find_match(card_hash, hash_dict)
        
        if match:
            # Get market data
            market_data = get_card_market_data(match['id'])
            
            return {
                "success": True,
                "card_id": match['id'],
                "name": match['name'],
                "match_confidence": (100 - (distance * 5)) if distance <= 10 else 0,
                "market_data": market_data
            }
        else:
            return {
                "success": False,
                "error": "No match found for this card",
                "best_distance": distance
            }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Error processing image: {str(e)}",
            "traceback": traceback.format_exc()
        }

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
