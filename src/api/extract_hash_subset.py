#!/usr/bin/env python3
"""
Script to extract a subset of the hash database for Docker image embedding.
This creates a smaller version of the hash database that can be included in the Docker image.
"""
import json
import os
import sys
import random

def extract_subset(input_file, output_file, num_entries=1000):
    """Extract a subset of entries from the hash database"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            hash_dict = json.load(f)
            
        print(f"Loaded hash database with {len(hash_dict)} entries")
        
        # If the database is smaller than requested subset, use all entries
        if len(hash_dict) <= num_entries:
            subset = hash_dict
        else:
            # Get a random subset of keys
            keys = list(hash_dict.keys())
            random.shuffle(keys)
            subset_keys = keys[:num_entries]
            subset = {k: hash_dict[k] for k in subset_keys}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(subset, f)
            
        print(f"Created subset with {len(subset)} entries at {output_file}")
        return True
    except Exception as e:
        print(f"Error creating subset: {e}")
        return False

if __name__ == "__main__":
    # Default paths
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(src_dir, "data", "hashes_dphash_16.json")
    output_file = os.path.join(src_dir, "api", "hash_subset.json")
    
    # Allow command line arguments to override defaults
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        num_entries = int(sys.argv[3])
    else:
        num_entries = 1000
    
    print(f"Extracting {num_entries} entries from {input_file} to {output_file}")
    success = extract_subset(input_file, output_file, num_entries)
    sys.exit(0 if success else 1)
