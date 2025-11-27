import os
import requests
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore


API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ygo-cards")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def cards_to_dataframe(cards) -> pd.DataFrame:
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
        })
    df = pd.DataFrame(rows)
    print(f"Built DataFrame with {len(df)} rows.")
    return df


def _safe_int(x):
    """Convert to int if possible, otherwise return None."""
    try:
        if pd.isna(x):
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def _safe_str(x):
    """Convert to str if not NaN/None, else return None."""
    if pd.isna(x):
        return None
    return str(x)


def build_documents(df: pd.DataFrame):
    docs: list[Document] = []

    for _, row in df.iterrows():
        # Cleaned fields
        name = _safe_str(row.get("name"))
        card_type = _safe_str(row.get("type"))
        race = _safe_str(row.get("race"))
        attribute = _safe_str(row.get("attribute"))
        level = _safe_int(row.get("level"))
        atk = _safe_int(row.get("atk"))
        defe = _safe_int(row.get("def"))
        archetype = _safe_str(row.get("archetype"))
        desc = _safe_str(row.get("desc")) or ""

        content_parts = [
            f"Name: {name or ''}",
            f"Type: {card_type or ''}",
            f"Race: {race or ''}",
            f"Attribute: {attribute or ''}",
            f"Level: {level if level is not None else ''}",
            f"ATK: {atk if atk is not None else ''}",
            f"DEF: {defe if defe is not None else ''}",
            f"Archetype: {archetype or ''}",
            "",
            f"Description: {desc}",
        ]
        content = "\n".join(content_parts)

        # Build metadata WITHOUT any None values
        metadata = {}

        card_id = _safe_int(row.get("id"))
        if card_id is not None:
            metadata["id"] = card_id

        if name is not None:
            metadata["name"] = name
        if card_type is not None:
            metadata["type"] = card_type
        if race is not None:
            metadata["race"] = race
        if attribute is not None:
            metadata["attribute"] = attribute
        if level is not None:
            metadata["level"] = level
        if atk is not None:
            metadata["atk"] = atk
        if defe is not None:
            metadata["def"] = defe
        if archetype is not None:
            metadata["archetype"] = archetype

        docs.append(Document(page_content=content, metadata=metadata))

    print(f"Built {len(docs)} documents for Pinecone.")
    return docs


def upload_to_pinecone(docs: list[Document]):
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Index name: {PINECONE_INDEX_NAME}")
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print("Finished uploading documents to Pinecone.")
    return vectorstore


def main():
    if not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Please set it in your environment or .env file."
        )

    cards = fetch_all_cards()
    df = cards_to_dataframe(cards)
    docs = build_documents(df)
    upload_to_pinecone(docs)
    print("Done. Pinecone index is now populated with Yu-Gi-Oh card chunks.")


if __name__ == "__main__":
    main()