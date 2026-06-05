"""
Build a ChromaDB vector store from website_likability/train_dataset.csv.

Structure:
  website_likability/vector_db/
    persona_1  ... persona_40   (one collection per persona_id)

Each document = image embedded with CLIP (clip-ViT-B-32).
Metadata = image path, ratings, respondent demographics, persona.

Inference helper:
    from build_vector_db_website_aes import query_similar
    results = query_similar(persona_id=3, image_path="path/to/image.png", k=5)
"""

import argparse
import os
import pandas as pd
import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

TRAIN_DATASET_PATH = os.path.join("website_likability", "train_dataset.csv")
VECTOR_DB_PATH     = os.path.join("website_likability", "vector_db")
IMAGE_BASE         = os.path.join(
    "website_likability",
    "website-aesthetics-datasets-master",
    "rating-based-dataset",
    "preprocess",
    "resized",
)
EMBEDDING_MODEL = "clip-ViT-B-32"
BATCH_SIZE      = 64   # smaller than text batches — images are heavier

METADATA_COLUMNS = [
    "row_id", "image", "age", "gender",
    "mean_response", "std_response", "difference_response",
    "persona_description",
]


def _get_client():
    return chromadb.PersistentClient(path=VECTOR_DB_PATH)


def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def _sanitize_metadata(records: list[dict]) -> list[dict]:
    """Ensure all metadata values are ChromaDB-compatible primitives."""
    sanitized = []
    for rec in records:
        sanitized.append({
            k: v if isinstance(v, (str, int, float, bool)) else str(v)
            for k, v in rec.items()
        })
    return sanitized


def _image_path(relative: str) -> str:
    """Convert the stored relative image path to a full filesystem path."""
    return os.path.join(IMAGE_BASE, relative.lstrip("/").replace("/", os.sep))


def _encode_images(model: SentenceTransformer, paths: list[str]) -> tuple[list, list[int]]:
    """Load and encode images; skip files that can't be opened.

    Returns (embeddings, valid_indices) where valid_indices are the positions
    in `paths` that were successfully loaded.
    """
    images, valid = [], []
    for i, p in enumerate(paths):
        try:
            images.append(Image.open(p).convert("RGB"))
            valid.append(i)
        except Exception:
            pass
    if not images:
        return [], []
    embeddings = model.encode(images, batch_size=BATCH_SIZE, show_progress_bar=False)
    return embeddings.tolist(), valid


def build_vector_db(input_path: str = TRAIN_DATASET_PATH, reset: bool = True, n_limit: int | None = None) -> None:
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if n_limit is not None:
        df = df.iloc[:n_limit]
        print(f"Using first {len(df):,} rows (--n-limit {n_limit})")

    for col in METADATA_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    print(f"Loading CLIP model: {EMBEDDING_MODEL}")
    model  = _load_model()
    client = _get_client()

    persona_ids = sorted(df["persona_id"].unique())
    print(f"Found {len(persona_ids)} personas — building {len(persona_ids)} collections.\n")

    total_docs = 0
    skipped    = 0

    with tqdm(total=len(df), desc="Embedding", unit="img") as pbar:
        for persona_id in persona_ids:
            collection_name = f"persona_{int(persona_id)}"
            pbar.set_postfix(persona=collection_name)

            if reset:
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass

            # No embedding_function — embeddings are provided directly
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            group = df[df["persona_id"] == persona_id].reset_index(drop=True)

            for start in range(0, len(group), BATCH_SIZE):
                batch      = group.iloc[start:start + BATCH_SIZE]
                img_paths  = [_image_path(p) for p in batch["image"]]
                embeddings, valid = _encode_images(model, img_paths)

                if not embeddings:
                    skipped += len(batch)
                    pbar.update(len(batch))
                    continue

                valid_batch = batch.iloc[valid].reset_index(drop=True)
                collection.upsert(
                    ids=[f"{int(persona_id)}_{start + v}" for v in valid],
                    embeddings=embeddings,
                    documents=valid_batch["image"].tolist(),
                    metadatas=_sanitize_metadata(
                        valid_batch[METADATA_COLUMNS].to_dict("records")
                    ),
                )
                skipped    += len(batch) - len(valid)
                total_docs += len(valid)
                pbar.update(len(batch))

    print(f"\nDone. {total_docs:,} documents embedded across {len(persona_ids)} collections.")
    if skipped:
        print(f"Skipped {skipped:,} rows (image file not found).")
    print(f"Vector DB saved to: {VECTOR_DB_PATH}")


def query_similar(persona_id: int, image_path: str, k: int = 5) -> list[dict]:
    """Return the top-k most similar training images for a given persona.

    Args:
        persona_id: 1-indexed persona ID.
        image_path: Path to the query image file.
        k: Number of results to return.

    Returns:
        List of dicts with image path, distance, and metadata fields.
    """
    model      = _load_model()
    client     = _get_client()
    collection = client.get_collection(name=f"persona_{persona_id}")

    img       = Image.open(image_path).convert("RGB")
    embedding = model.encode(img).tolist()

    results = collection.query(query_embeddings=[embedding], n_results=k)
    return [
        {"image": doc, "distance": dist, **meta}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector store from website aesthetics training data."
    )
    parser.add_argument(
        "--input", default=TRAIN_DATASET_PATH,
        help=f"Path to training CSV (default: {TRAIN_DATASET_PATH}).",
    )
    parser.add_argument(
        "--no-reset", action="store_false", dest="reset",
        help="Upsert into existing collections instead of wiping and rebuilding.",
    )
    parser.set_defaults(reset=True)
    parser.add_argument(
        "--n-limit", type=int, default=None, metavar="N",
        help="Use only the first N rows of the training CSV (default: all).",
    )
    args = parser.parse_args()
    build_vector_db(input_path=args.input, reset=args.reset, n_limit=args.n_limit)
