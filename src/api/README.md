 # Card Scanner API for n8n Integration

This API allows you to integrate the Pokémon card scanner with your n8n workflow for analyzing eBay listings.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure your `credentials.json` file is in the project root with your PokemonTCG.io API key:
   ```json
   {
     "api_key": "your-api-key-here"
   }
   ```

3. Start the API server:
   ```
   python card_api.py
   ```

## API Endpoints

### Identify Cards

**Endpoint:** `/identify`
**Method:** POST
**Description:** Identifies Pokémon cards in an image and returns card information with market data.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "cards": [
    {
      "card_index": 0,
      "card_id": "sv5-1",
      "name": "Scyther",
      "match_confidence": 90,
      "market_data": {
        "id": "sv5-1",
        "name": "Scyther",
        "set": "Scarlet & Violet—Paldean Fates",
        "rarity": "Common",
        "market_data": {
          "tcgplayer": {
            "normal": {
              "low": 0.05,
              "mid": 0.15,
              "high": 1.0,
              "market": 0.08,
              "directLow": 0.05
            }
          },
          "cardmarket": {
            "averageSellPrice": 0.06,
            "lowPrice": 0.02,
            "trendPrice": 0.07,
            "germanProLow": null,
            "suggestedPrice": null,
            "reverseHoloSell": null,
            "reverseHoloLow": null,
            "reverseHoloTrend": null,
            "lowPriceExPlus": 0.02,
            "avg1": 0.06,
            "avg7": 0.07,
            "avg30": 0.08,
            "reverseHoloAvg1": null,
            "reverseHoloAvg7": null,
            "reverseHoloAvg30": null
          }
        }
      }
    }
  ]
}
```

### Get Market Data for a Specific Card

**Endpoint:** `/market-data/{card_id}`
**Method:** GET
**Description:** Returns market data for a specific card ID.

**Response:**
```json
{
  "id": "sv5-1",
  "name": "Scyther",
  "set": "Scarlet & Violet—Paldean Fates",
  "rarity": "Common",
  "market_data": {
    "tcgplayer": {
      "normal": {
        "low": 0.05,
        "mid": 0.15,
        "high": 1.0,
        "market": 0.08,
        "directLow": 0.05
      }
    },
    "cardmarket": {
      "averageSellPrice": 0.06,
      "lowPrice": 0.02,
      "trendPrice": 0.07,
      "germanProLow": null,
      "suggestedPrice": null,
      "reverseHoloSell": null,
      "reverseHoloLow": null,
      "reverseHoloTrend": null,
      "lowPriceExPlus": 0.02,
      "avg1": 0.06,
      "avg7": 0.07,
      "avg30": 0.08,
      "reverseHoloAvg1": null,
      "reverseHoloAvg7": null,
      "reverseHoloAvg30": null
    }
  }
}
```

## n8n Integration

To integrate with your n8n workflow:

1. Use the n8n HTTP Request node to call the `/identify` endpoint
2. Convert your eBay images to base64 format
3. Send the base64-encoded image in the request body
4. Process the response to extract card information and market data

### Example n8n Workflow:

1. **eBay Trigger Node** → Triggers when new listings are found
2. **HTTP Request Node** → Downloads the image from eBay
3. **Function Node** → Converts the image to base64
4. **HTTP Request Node** → Sends the image to your Card Scanner API
5. **Function Node** → Processes the response and extracts card data
6. **PokemonTCG API Node** → (Optional) Get additional card data if needed
7. **Action Node** → Take action based on the card data (e.g., save to database, send notification)

## Deployment

For production use, consider deploying this API using Gunicorn:

```
gunicorn -w 4 -b 0.0.0.0:5000 card_api:app
```

You can also containerize the API using Docker for easier deployment.
