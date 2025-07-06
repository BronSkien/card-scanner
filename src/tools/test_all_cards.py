import os
import json
import imagehash
from PIL import Image
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
hash_size = 16  # bytes - must match the scanner.py setting
input_folder = "../../card_images"  # Path to card images folder
hash_db_file = "../data/hashes_dphash_16.json"  # Path to the hash database
results_file = "../../card_test_results.json"  # Path to save test results
max_workers = 8  # Number of parallel workers

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
    
    for card_id, card_data in hash_dict.items():
        card_hash = card_data.get('hash')
        if card_hash:
            distance = hamming_distance(hash_value, card_hash)
            if distance < best_distance:
                best_distance = distance
                best_match = card_data
    
    if best_match and best_distance <= threshold:
        return best_match, best_distance
    else:
        return None, best_distance

def parse_card_info(filepath, filename):
    """Parse card ID from filepath and filename"""
    # Get the folder name (set) and filename (card number)
    folder_path = os.path.dirname(filepath)
    set_code = os.path.basename(folder_path)
    
    # Remove file extension from filename to get card number
    card_number = os.path.splitext(filename)[0]
    
    # If the file is directly in the card_images folder (no subfolder)
    if set_code == "card_images":
        # Try to extract set code and card number from the filename
        # Common format examples: swsh1-23, sv01-123, sm12-234, etc.
        import re
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

def test_card(args):
    """Test a single card image against the hash database"""
    file_path, filename, hash_dict, threshold = args
    
    # Parse set code and card number from filepath and filename
    set_code, card_number = parse_card_info(file_path, filename)
    expected_id = f"{set_code}-{card_number}"
    
    # Generate hash for the image
    hash_value = hash_image(file_path)
    if not hash_value:
        return {
            "file_path": file_path,
            "expected_id": expected_id,
            "status": "error",
            "message": "Failed to generate hash"
        }
    
    # Find the best match
    best_match, distance = find_match(hash_value, hash_dict, threshold)
    
    if best_match:
        match_id = best_match.get('id')
        match_name = best_match.get('name')
        
        # Check if the match is correct (matches the expected ID)
        is_correct = match_id == expected_id
        
        return {
            "file_path": file_path,
            "expected_id": expected_id,
            "match_id": match_id,
            "match_name": match_name,
            "distance": distance,
            "is_correct": is_correct,
            "status": "matched" if is_correct else "mismatched"
        }
    else:
        return {
            "file_path": file_path,
            "expected_id": expected_id,
            "status": "no_match",
            "distance": distance
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test all card images against the hash database')
    parser.add_argument('--threshold', type=int, default=10, help='Hamming distance threshold for matches')
    parser.add_argument('--limit', type=int, default=0, help='Limit the number of cards to test (0 for all)')
    parser.add_argument('--set', type=str, default='', help='Test only a specific set')
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # Load hash database
    if not os.path.exists(hash_db_file):
        print(f"Error: Hash database {hash_db_file} does not exist")
        return
    
    print(f"Loading hash database from {hash_db_file}...")
    with open(hash_db_file, 'r', encoding='utf-8') as json_file:
        hash_dict = json.load(json_file)
    
    print(f"Loaded hash database with {len(hash_dict)} entries")
    
    # Find all card images
    print(f"Scanning {input_folder} for card images...")
    image_files = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_folder):
        # Skip the README.md file
        if os.path.basename(root) == "card_images" and "__pycache__" in dirs:
            dirs.remove("__pycache__")
        
        # If a specific set is specified, only process that set
        if args.set and os.path.basename(root) != args.set and root != input_folder:
            continue
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file != "README.md":
                # Store full path and filename
                image_files.append((os.path.join(root, file), file))
    
    # Limit the number of cards to test if specified
    if args.limit > 0:
        image_files = image_files[:args.limit]
    
    print(f"Found {len(image_files)} card images to test")
    
    # Test all card images
    results = []
    test_args = [(file_path, filename, hash_dict, args.threshold) for file_path, filename in image_files]
    
    # Use ThreadPoolExecutor for parallel processing
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test_card, arg) for arg in test_args]
        
        # Show progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Testing cards"):
            result = future.result()
            results.append(result)
    
    end_time = time.time()
    
    # Analyze results
    total_cards = len(results)
    matched_cards = sum(1 for r in results if r['status'] == 'matched')
    mismatched_cards = sum(1 for r in results if r['status'] == 'mismatched')
    no_match_cards = sum(1 for r in results if r['status'] == 'no_match')
    error_cards = sum(1 for r in results if r['status'] == 'error')
    
    # Calculate match rate
    match_rate = matched_cards / total_cards * 100 if total_cards > 0 else 0
    
    # Print summary
    print("\n===== Test Results =====")
    print(f"Total cards tested: {total_cards}")
    print(f"Correctly matched: {matched_cards} ({match_rate:.2f}%)")
    print(f"Mismatched: {mismatched_cards}")
    print(f"No match found: {no_match_cards}")
    print(f"Errors: {error_cards}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # List some mismatched cards if any
    if mismatched_cards > 0:
        print("\n===== Sample Mismatches =====")
        mismatches = [r for r in results if r['status'] == 'mismatched']
        for i, mismatch in enumerate(mismatches[:10]):  # Show up to 10 mismatches
            print(f"{i+1}. {mismatch['file_path']}")
            print(f"   Expected: {mismatch['expected_id']}")
            print(f"   Matched as: {mismatch['match_id']} ({mismatch['match_name']})")
            print(f"   Distance: {mismatch['distance']}")
    
    # Save results to file
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_cards": total_cards,
                "matched_cards": matched_cards,
                "mismatched_cards": mismatched_cards,
                "no_match_cards": no_match_cards,
                "error_cards": error_cards,
                "match_rate": match_rate,
                "time_taken": end_time - start_time
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")

if __name__ == "__main__":
    main()
