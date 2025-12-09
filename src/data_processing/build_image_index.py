import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PIL import Image
from pinecone import Pinecone

load_dotenv()

CSV_PATH = Path("data/images/cards_index.csv")
IMAGE_ROOT = Path(".")  # image_path in CSV is relative to project root

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_IMAGE_INDEX_NAME = os.environ.get("PINECONE_IMAGE_INDEX_NAME", "ygo-card-images")

CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"


def load_index_df() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if "image_path" not in df.columns or "name" not in df.columns:
        raise ValueError("CSV must have at least 'image_path' and 'name' columns.")
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    return df


def embed_images(df: pd.DataFrame, model: SentenceTransformer):
    vectors = []
    ids = []
    metadatas = []

    for idx, row in df.iterrows():
        rel_path = row["image_path"]
        name = str(row["name"])
        img_path = (IMAGE_ROOT / rel_path).resolve()

        if not img_path.exists():
            print(f"[WARN] Image not found, skipping: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open image {img_path}: {e}")
            continue

        emb = model.encode(image, convert_to_numpy=True, normalize_embeddings=True)

        vector_id = f"{idx}"
        metadata = {
            "name": name,
            "image_path": rel_path,
        }

        ids.append(vector_id)
        vectors.append(emb.tolist())
        metadatas.append(metadata)

    print(f"Prepared {len(vectors)} image embeddings.")
    return ids, vectors, metadatas


def upload_to_pinecone(ids, vectors, metadatas):
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_IMAGE_INDEX_NAME)

    # Upsert in batches
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        batch = list(
            zip(ids[start:end], vectors[start:end], metadatas[start:end])
        )
        print(f"Upserting batch {start}–{end}...")
        index.upsert(vectors=batch)

    print("Finished uploading image vectors to Pinecone.")


def main():
    df = load_index_df()
    print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
    model = SentenceTransformer(CLIP_MODEL_NAME)

    ids, vectors, metadatas = embed_images(df, model)
    upload_to_pinecone(ids, vectors, metadatas)
    print("Done. Image index populated.")


if __name__ == "__main__":
    main()