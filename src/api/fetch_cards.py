import json
import requests
import pandas as pd


API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

RAW_JSON_PATH = "data/raw/cards_raw.json"
FLAT_CSV_PATH = "data/processed/cards_flat.csv"


def fetch_all_cards():
    """Fetch all Yu-Gi-Oh cards from the YGOPRODeck API."""
    print(f"Requesting data from {API_URL} ...")
    resp = requests.get(API_URL, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"API request failed with status {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    if "data" not in data:
        raise ValueError("Unexpected API response format: missing 'data' field")

    cards = data["data"]
    print(f"Fetched {len(cards)} cards from API.")
    return cards


def save_raw_json(cards):
    """Save the full card list as raw JSON."""
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    print(f"Saved raw card data to {RAW_JSON_PATH}")


def flatten_cards_to_dataframe(cards):
    """Flatten key card fields into a DataFrame for easier processing."""
    rows = []
    for c in cards:
        rows.append({
            "id": c.get("id"),
            "name": c.get("name"),
            "type": c.get("type"),
            "frameType": c.get("frameType"),
            "race": c.get("race"),
            "attribute": c.get("attribute"),
            "level": c.get("level"),
            "atk": c.get("atk"),
            "def": c.get("def"),
            "archetype": c.get("archetype"),
            "desc": c.get("desc"),
            "image_url": c.get("card_images", [{}])[0].get("image_url"),
        })

    return pd.DataFrame(rows)


def save_flat_csv(df):
    """Save the flattened card info as CSV."""
    df.to_csv(FLAT_CSV_PATH, index=False)
    print(f"Saved flattened card data to {FLAT_CSV_PATH}")


def main():
    cards = fetch_all_cards()
    save_raw_json(cards)
    df = flatten_cards_to_dataframe(cards)
    save_flat_csv(df)
    print("Done. Local Yu-Gi-Oh card dataset ready!")


if __name__ == "__main__":
    main()