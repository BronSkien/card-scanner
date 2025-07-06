# Adding Your Card Images to the Scanner

This directory is where you should place your high-resolution PNG card images to be added to the card scanner database.

## Organization Methods

You can organize your card images in two ways:

### Method 1: Using Folder Structure (Recommended)

Organize your cards in folders by set, with each card's filename being its number:

```
card_images/
├── base1/
│   ├── 1.png  (Alakazam)
│   ├── 2.png  (Blastoise)
│   └── 4.png  (Charizard)
├── swsh1/
│   ├── 23.png
│   └── 24.png
└── sv01/
    ├── 123.png
    └── 124.png
```

With this structure, the script will use the folder name as the set code and the filename (without extension) as the card number. For example, `card_images/base1/1.png` will be identified as card `base1-1` (Base Set Alakazam).

### Method 2: Using Filename Convention

Alternatively, you can place all files directly in the `card_images` folder with filenames in this format:

```
[set_code]-[card_number].png
```

Examples:
- `swsh1-23.png` (Sword & Shield base set, card #23)
- `sv01-123.png` (Scarlet & Violet base set, card #123)
- `sm12-234.png` (Sun & Moon Cosmic Eclipse, card #234)

Both methods allow the script to automatically look up card information from the PokemonTCG.io API using the set code and card number.

## Instructions

1. **Prepare Your Images**
   - Place all your high-resolution PNG card images in this directory
   - Name files according to the convention above
   - Make sure images are clean and clearly show the entire card

2. **Run the Enhanced Hash Generation Script with API Integration**
   ```bash
   cd c:\Projects\card_scanner\card-scanner\src\tools
   python generate_hashes_with_api.py
   ```
   
   This script will:
   - Generate hashes for each card image
   - Look up card information from the PokemonTCG.io API
   - Add detailed card names and metadata to the hash database

3. **Check the Results**
   - The script will output how many new cards were added to the database
   - It will also show how many cards were enhanced with API data
   - Progress is saved every 50 entries to prevent data loss

## Important Notes

- The script will skip any card IDs that already exist in the database
- API lookups require a valid PokemonTCG.io API key in `credentials.json`
- If the API lookup fails, the script will still add the card using the filename as the card name
- For best results, ensure your images are:
  - Clean, well-lit photos of the cards
  - Cropped to show only the card (no background)
  - In portrait orientation
  - High resolution but not excessively large (1000-2000px height is ideal)

## Alternative Method (Without API)

If you don't want to use the PokemonTCG.io API, you can still use the basic hash generation script:

```bash
cd c:\Projects\card_scanner\card-scanner\src\tools
python generate_hashes.py
```

This version doesn't require an API key but won't include detailed card information.
