import os
import json
import imagehash
from PIL import Image
import cv2
import numpy as np
import re
import time
import pokemontcgsdk
from pokemontcgsdk import Card
from pokemontcgsdk import RestClient

# Configuration
hash_size = 16  # bytes - must match the scanner.py setting
input_folder = "../../card_images"  # Change this to your PNG images folder path
output_hash_file = "../data/hashes_dphash_16.json"  # Path to the existing hash database
temp_hash_file = "../data/temp_hashes.json"  # Temporary file for new hashes
credentials_file = "../../credentials.json"  # Path to credentials file

# Load API key
try:
    with open(credentials_file, 'r') as f:
        credentials = json.load(f)
        api_key = credentials.get('api_key')
        if api_key:
            RestClient.configure(api_key)
            print(f"Successfully configured PokemonTCG API client")
        else:
            print("Warning: No API key found in credentials.json")
except Exception as e:
    print(f"Warning: Could not load API credentials: {e}")
    api_key = None

# Create hash for an image
def hash_image(img_path):
    # Open the image
    img = Image.open(img_path)
    
    # Convert to RGB (in case it's RGBA with transparency)
    img = img.convert('RGB')
    
    # Compute the hash
    dhash = imagehash.dhash(img, hash_size)
    phash = imagehash.phash(img, hash_size)
    
    # Combine the hashes
    hash_value = f'{dhash}{phash}'
    
    return hash_value

# Parse card ID from filepath, considering folder structure
def parse_card_info(filepath, filename):
    # Get the folder name (set) and filename (card number)
    folder_path = os.path.dirname(filepath)
    set_code = os.path.basename(folder_path)
    
    # Remove file extension from filename to get card number
    card_number = os.path.splitext(filename)[0]
    
    # If the file is directly in the card_images folder (no subfolder)
    if set_code == "card_images":
        # Try to extract set code and card number from the filename
        # Common format examples: swsh1-23, sv01-123, sm12-234, etc.
        base_name = os.path.splitext(filename)[0]
        match = re.match(r'([a-zA-Z]+\d+)-(\d+[a-zA-Z]*)', base_name)
        
        if match:
            set_code = match.group(1)
            card_number = match.group(2)
        else:
            # If no match, use the filename as both set and number
            set_code = base_name
            card_number = base_name
    
    return set_code, card_number

# Get card info from PokemonTCG API
def get_card_info(set_code, card_number):
    if not api_key:
        return None
    
    try:
        # Query the API for the card
        query = f'set.id:{set_code} number:{card_number}'
        cards = Card.where(q=query)
        
        # If we found a match, return the card info
        if cards and len(cards) > 0:
            card = cards[0]
            return {
                "name": card.name,
                "set": card.set.name,
                "rarity": getattr(card, 'rarity', 'Unknown'),
                "id": f"{set_code}-{card_number}"
            }
        else:
            print(f"No card found for {set_code}-{card_number}")
            return None
    except Exception as e:
        print(f"Error querying API for {set_code}-{card_number}: {e}")
        # Sleep to avoid rate limiting
        time.sleep(0.5)
        return None

def main():
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist. Please create it and add your card images.")
        return
    
    # Load existing hash database if it exists
    if os.path.exists(output_hash_file):
        with open(output_hash_file, 'r', encoding='utf-8') as json_file:
            hash_dict = json.load(json_file)
        print(f"Loaded existing hash database with {len(hash_dict)} entries.")
    else:
        hash_dict = {}
        print("No existing hash database found. Creating a new one.")
    
    # Process all images in the input folder and its subfolders
    new_entries = 0
    skipped_entries = 0
    api_entries = 0
    image_files = []
    
    # Walk through all directories and subdirectories
    print(f"Scanning {input_folder} and its subfolders for card images...")
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Store full path and filename
                image_files.append((os.path.join(root, file), file))
    
    print(f"Found {len(image_files)} card images to process")
    
    for i, (file_path, filename) in enumerate(image_files):
        # Parse set code and card number from filepath and filename
        set_code, card_number = parse_card_info(file_path, filename)
        card_id = f"{set_code}-{card_number}"
        
        # Skip if this card ID already exists
        if card_id in hash_dict:
            print(f"Skipping {card_id} - already in database")
            skipped_entries += 1
            continue
        
        # Generate hash
        try:
            hash_value = hash_image(file_path)
            
            # Try to get card info from API
            card_info = get_card_info(set_code, card_number)
            
            if card_info:
                # Use API info
                hash_dict[card_id] = {
                    "id": card_id,
                    "name": f"{card_info['name']} ({card_info['set']})",
                    "hash": hash_value,
                    "rarity": card_info['rarity']
                }
                api_entries += 1
            else:
                # Use filename as fallback
                hash_dict[card_id] = {
                    "id": card_id,
                    "name": card_id,
                    "hash": hash_value
                }
            
            new_entries += 1
            print(f"[{i+1}/{len(image_files)}] Added hash for {card_id}")
            
            # Save progress every 50 entries
            if new_entries % 50 == 0:
                with open(temp_hash_file, 'w', encoding='utf-8') as json_file:
                    json.dump(hash_dict, json_file)
                print(f"Progress saved: {new_entries} entries processed")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save the updated hash dictionary
    if new_entries > 0:
        # First save to a temporary file as a backup
        with open(temp_hash_file, 'w', encoding='utf-8') as json_file:
            json.dump(hash_dict, json_file)
        
        # Then save to the actual output file
        with open(output_hash_file, 'w', encoding='utf-8') as json_file:
            json.dump(hash_dict, json_file)
        
        print(f"\nSummary:")
        print(f"- Added {new_entries} new entries to the hash database")
        print(f"- {api_entries} entries were enhanced with API data")
        print(f"- {skipped_entries} entries were skipped (already in database)")
        print(f"- Total entries in the hash database: {len(hash_dict)}")
    else:
        print("No new entries were added to the hash database.")

if __name__ == "__main__":
    main()
