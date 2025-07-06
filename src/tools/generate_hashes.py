import os
import json
import imagehash
from PIL import Image
import cv2
import numpy as np

# Configuration
hash_size = 16  # bytes - must match the scanner.py setting
input_folder = "../../card_images"  # Change this to your PNG images folder path
output_hash_file = "../data/hashes_dphash_16.json"  # Path to the existing hash database
temp_hash_file = "../data/temp_hashes.json"  # Temporary file for new hashes

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
    
    # Process all images in the input folder
    new_entries = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            
            # Generate a card ID from the filename (remove extension)
            card_id = os.path.splitext(filename)[0]
            
            # Skip if this card ID already exists
            if card_id in hash_dict:
                print(f"Skipping {card_id} - already in database")
                continue
            
            # Generate hash
            try:
                hash_value = hash_image(file_path)
                
                # Add to hash dictionary
                hash_dict[card_id] = {
                    "id": card_id,
                    "name": card_id,  # You might want to use a more descriptive name
                    "hash": hash_value
                }
                
                new_entries += 1
                print(f"Added hash for {card_id}")
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
        
        print(f"Added {new_entries} new entries to the hash database.")
        print(f"Total entries in the hash database: {len(hash_dict)}")
    else:
        print("No new entries were added to the hash database.")

if __name__ == "__main__":
    main()
