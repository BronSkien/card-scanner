import os
import json
import imagehash
from PIL import Image
import argparse

# Configuration
hash_size = 16  # bytes - must match the scanner.py setting
hash_db_file = "../data/hashes_dphash_16.json"  # Path to the hash database

def hash_image(img_path):
    """Generate hash for an image"""
    try:
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
    except Exception as e:
        print(f"Error hashing image {img_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Calculate the Hamming distance between two hashes"""
    if len(hash1) != len(hash2):
        raise ValueError("Hashes must be of equal length")
    
    # Count the number of differing bits
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def find_match(hash_value, hash_dict, threshold=10):
    """Find the best match for a hash in the hash dictionary"""
    best_match = None
    best_distance = float('inf')
    top_matches = []
    
    print(f"Looking for matches with threshold {threshold}...")
    
    for card_id, card_data in hash_dict.items():
        card_hash = card_data.get('hash')
        if card_hash:
            distance = hamming_distance(hash_value, card_hash)
            
            # Keep track of top 5 matches
            if len(top_matches) < 5:
                top_matches.append((card_data, distance))
                top_matches.sort(key=lambda x: x[1])
            elif distance < top_matches[-1][1]:
                top_matches[-1] = (card_data, distance)
                top_matches.sort(key=lambda x: x[1])
                
            if distance < best_distance:
                best_distance = distance
                best_match = card_data
    
    print("\nTop 5 potential matches:")
    for card, dist in top_matches:
        print(f"- {card.get('id')}: {card.get('name')} (distance: {dist})")
    
    if best_match and best_distance <= threshold:
        return best_match, best_distance
    else:
        return None, best_distance

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Verify card hash matching')
    parser.add_argument('--image', required=True, help='Path to the card image to verify')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} does not exist")
        return
    
    # Load hash database
    if not os.path.exists(hash_db_file):
        print(f"Error: Hash database {hash_db_file} does not exist")
        return
    
    with open(hash_db_file, 'r', encoding='utf-8') as json_file:
        hash_dict = json.load(json_file)
    
    print(f"Loaded hash database with {len(hash_dict)} entries")
    
    # Generate hash for the input image
    hash_value = hash_image(args.image)
    if not hash_value:
        print("Failed to generate hash for the input image")
        return
    
    print(f"Generated hash for {args.image}: {hash_value}")
    
    # Find the best match
    best_match, distance = find_match(hash_value, hash_dict)
    
    if best_match:
        print(f"\nFound match with distance {distance}:")
        print(f"Card ID: {best_match.get('id')}")
        print(f"Card Name: {best_match.get('name')}")
        if 'rarity' in best_match:
            print(f"Rarity: {best_match.get('rarity')}")
    else:
        print(f"\nNo match found. Best distance was {distance}, which exceeds the threshold.")

if __name__ == "__main__":
    main()
