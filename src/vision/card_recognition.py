import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

import easyocr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ygo-cards")


def _build_vectorstore() -> PineconeVectorStore:
    """Reuse the same Pinecone index we built before."""
    if not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY is not set.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


def _build_ocr_reader() -> easyocr.Reader:
    """Initialize EasyOCR reader (English only, CPU)."""
    # gpu=False keeps it simple on most machines
    return easyocr.Reader(['en', 'de'], gpu=False)

_vectorstore = _build_vectorstore()
_ocr_reader = _build_ocr_reader()

def ocr_card_name(image_path: str, reader: easyocr.Reader = _ocr_reader) -> Tuple[str, List[str]]:
    """
    Run OCR on the image and try to guess the card name text.
    We assume the card name is usually near the top of the card.
    """
    image_path = str(image_path)
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # detail=1 → we get bounding boxes, text, confidence
    results = reader.readtext(image_path, detail=1)

    if not results:
        raise ValueError("No text detected in image.")

    # Each result: (bbox, text, conf)
    # bbox is list of 4 points: [top-left, top-right, bottom-right, bottom-left]
    # We'll pick the text with the smallest average Y (closest to top of image)
    candidates: List[Tuple[float, str]] = []
    all_texts: List[str] = []

    for bbox, text, _ in results:
        all_texts.append(text)
        # bbox: list of 4 (x, y) points → compute average y
        ys = [pt[1] for pt in bbox]
        avg_y = sum(ys) / len(ys)
        candidates.append((avg_y, text))

    # Sort by vertical position (top first)
    candidates.sort(key=lambda x: x[0])

    # Best guess: text from the top-most detected box
    best_text = candidates[0][1].strip()

    print(f"OCR raw texts: {all_texts}")
    print(f"Guessed card name from OCR: {best_text!r}")

    return best_text, all_texts


def resolve_card_from_text(
    ocr_text: str,
    k: int = 3
) -> Tuple[Document, List[Document]]:
    """
    Use Pinecone similarity search on the OCR text to find the closest card.
    Because our stored docs contain 'Name: <card name>' in the content,
    querying with the OCR'd name usually returns the right card on top.
    """
    # similarity_search returns a list[Document]
    query_text = f"Name: {ocr_text}"
    docs = _vectorstore.similarity_search(query_text, k=k)

    if not docs:
        raise ValueError("No matching cards found in vector store.")

    best = docs[0]
    return best, docs


def identify_card_from_image(image_path: str):
    """
    Full pipeline:
    - OCR the card name from the image
    - Use Pinecone to resolve to the most likely Yu-Gi-Oh card
    """
    print(f"Identifying card from image: {image_path}")

    ocr_guess, _ = ocr_card_name(image_path, _ocr_reader)
    best_doc, docs = resolve_card_from_text(ocr_guess, k=5)

    meta = best_doc.metadata or {}
    card_name = meta.get("name", "Unknown")
    card_type = meta.get("type", "Unknown type")

    print(f"\nPredicted card: {card_name} ({card_type})")
    print("\nTop candidates:")
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        print(f"{i}. {m.get('name', 'Unknown')} ({m.get('type', 'Unknown type')})")

    return card_name, docs


def main():
    print("Yu-Gi-Oh Card Image Recognition (OCR + Pinecone)")
    path = input("Path to card image: ").strip()
    if not path:
        print("No path provided, exiting.")
        return

    identify_card_from_image(path)


if __name__ == "__main__":
    main()