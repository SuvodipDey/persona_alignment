"""
Build a ChromaDB vector store from opinion_qa/train_dataset.csv.

Structure:
  opinion_qa/vector_db/
    persona_1  ... persona_40   (one collection per persona_id)

Each document = the `query` field (embedded).
Metadata = all other columns (for RAG retrieval).

Inference helper:
    from build_vector_db_opinion_qa import query_similar
    results = query_similar(persona_id=3, query="...", k=5)
"""

import argparse
import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

TRAIN_DATASET_PATH = os.path.join("opinion_qa", "train_dataset.csv")
VECTOR_DB_PATH     = os.path.join("opinion_qa", "vector_db")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2" #"BAAI/bge-base-en-v1.5"
BATCH_SIZE         = 500

METADATA_COLUMNS = [
    "survey", "question_key", "options", "option_ordinal",
    "human_response", "attributes", "persona_description",
    "persona_matching", "unmatched_attributes",
]


def _get_client_and_ef():
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client, ef


def _sanitize_metadata(records: list[dict]) -> list[dict]:
    """Ensure all metadata values are ChromaDB-compatible primitives."""
    sanitized = []
    for rec in records:
        sanitized.append({
            k: v if isinstance(v, (str, int, float, bool)) else str(v)
            for k, v in rec.items()
        })
    return sanitized


def build_vector_db(input_path: str = TRAIN_DATASET_PATH, reset: bool = True, n_limit: int | None = None) -> None:
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if n_limit is not None:
        df = df.iloc[:n_limit]
        print(f"Using first {len(df):,} rows (--n-limit {n_limit})")

    # Fill nulls in metadata columns so ChromaDB never receives NaN
    for col in METADATA_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    client, ef = _get_client_and_ef()

    persona_ids = sorted(df["persona_id"].unique())
    print(f"Found {len(persona_ids)} personas — building {len(persona_ids)} collections.\n")

    total_docs = 0
    with tqdm(total=len(df), desc="Embedding", unit="doc") as pbar:
        for persona_id in persona_ids:
            collection_name = f"persona_{int(persona_id)}"
            pbar.set_postfix(persona=collection_name)

            if reset:
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass

            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )

            group = df[df["persona_id"] == persona_id].reset_index(drop=True)

            for start in range(0, len(group), BATCH_SIZE):
                batch = group.iloc[start:start + BATCH_SIZE]
                collection.upsert(
                    ids=[f"{int(persona_id)}_{i + start}" for i in range(len(batch))],
                    documents=batch["query"].tolist(),
                    metadatas=_sanitize_metadata(
                        batch[METADATA_COLUMNS].to_dict("records")
                    ),
                )
                pbar.update(len(batch))

            total_docs += len(group)

    print(f"\nDone. {total_docs:,} total documents across {len(persona_ids)} collections.")
    print(f"Vector DB saved to: {VECTOR_DB_PATH}")


def query_similar(persona_id: int, query: str, k: int = 5) -> list[dict]:
    """Fetch the top-k most similar training records for a given persona and query.

    Returns a list of dicts, each containing the original query, all metadata
    fields, and a cosine distance score (lower = more similar).
    """
    client, ef = _get_client_and_ef()
    collection  = client.get_collection(
        name=f"persona_{persona_id}",
        embedding_function=ef,
    )
    results = collection.query(query_texts=[query], n_results=k)

    return [
        {"query": doc, "distance": dist, **meta}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector store from opinion-QA training data."
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
