from pathlib import Path
import csv
import random

from datasets import load_dataset
from PIL import Image


# Where to save subset images
SUBSET_DIR = Path("data/images/subset")
SUBSET_DIR.mkdir(parents=True, exist_ok=True)

# Where to write the CSV that build_image_index.py will use
CSV_PATH = Path("data/images/cards_index.csv")

# How many cards you want in the subset (change as you like)
NUM_CARDS = 40


def main():
    print("Loading Hugging Face dataset FabioArdi/yugioh_images...")
    ds = load_dataset("FabioArdi/yugioh_images", split="train")

    total = len(ds)
    print(f"Dataset has {total} rows.")

    # Randomly pick a subset of indices
    indices = list(range(total))
    random.shuffle(indices)
    subset_indices = indices[:NUM_CARDS]

    print(f"Sampling {len(subset_indices)} cards for the subset...")

    rows_for_csv = []

    for i, idx in enumerate(subset_indices, start=1):
        example = ds[idx]
        img: Image.Image = example["image"]
        name: str = example["name"]

        # Build a safe filename
        safe_name = (
            name.replace(" ", "_")
                .replace("/", "_")
                .replace(":", "_")
                .replace("?", "")
        )

        filename = f"{safe_name}_{idx}.jpg"
        out_path = SUBSET_DIR / filename

        # Save the image
        img.save(out_path, format="JPEG")

        # Path in CSV should be relative to project root
        rel_path = out_path.as_posix()

        rows_for_csv.append((rel_path, name))
        print(f"[{i}/{len(subset_indices)}] Saved {rel_path} -> {name}")

    # Write CSV
    print(f"Writing CSV to {CSV_PATH} ...")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "name"])
        for rel_path, name in rows_for_csv:
            writer.writerow([rel_path, name])

    print("Done. Subset images + cards_index.csv created.")


if __name__ == "__main__":
    main()