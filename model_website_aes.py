import argparse
import asyncio
import base64
import os
import random
import re
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm
import pandas as pd
import chromadb

from build_vector_db_website_aes import VECTOR_DB_PATH, EMBEDDING_MODEL

MAX_CONCURRENCY = 32
MAX_RETRIES     = 5
RETRY_BASE_SECS = 2
K_EXAMPLES      = 3

load_dotenv()

API_KEY  = os.getenv("OPENROUTER_KEY")
BASE_URL = os.getenv("OPENROUTER_URL")

MODEL        = "qwen/qwen3-vl-8b-instruct"
DATASET_PATH = "website_likability/test_dataset.csv"
IMAGE_BASE   = "website_likability/website-aesthetics-datasets-master/rating-based-dataset/preprocess/resized"
OUTPUT_PATH  = "website_likability/output_model.csv"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


_image_cache: dict[str, str] = {}

def load_image_b64(image_rel: str) -> str:
    if image_rel not in _image_cache:
        path = os.path.join(IMAGE_BASE, image_rel.lstrip("/"))
        with open(path, "rb") as f:
            _image_cache[image_rel] = base64.b64encode(f.read()).decode("utf-8")
    return _image_cache[image_rel]


# CLIP model and ChromaDB client — loaded once on first use.
_clip_model: SentenceTransformer | None = None
_chroma_client: chromadb.ClientAPI | None = None
_example_cache: dict[tuple, list[dict]] = {}


def _get_clip_model() -> SentenceTransformer:
    global _clip_model
    if _clip_model is None:
        print(f"Loading CLIP model: {EMBEDDING_MODEL}")
        _clip_model = SentenceTransformer(EMBEDDING_MODEL)
    return _clip_model


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    return _chroma_client


def fetch_examples(persona_id: int, image_rel: str, k: int) -> list[dict]:
    """Return top-k visually similar training images for a given persona and query image."""
    cache_key = (persona_id, image_rel, k)
    if cache_key in _example_cache:
        return _example_cache[cache_key]

    try:
        model      = _get_clip_model()
        collection = _get_chroma_client().get_collection(name=f"persona_{persona_id}")

        img_path  = os.path.join(IMAGE_BASE, image_rel.lstrip("/"))
        embedding = model.encode(Image.open(img_path).convert("RGB")).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=k + 1,
            where={"image": {"$ne": image_rel}},
        )
        examples = [
            {
                "image":         doc,
                "mean_response": meta["mean_response"],
                "image_b64":     load_image_b64(doc),
            }
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            )
        ][:k]
    except Exception as e:
        print(f"  [fetch_examples] persona={persona_id} image={image_rel}: {e}")
        examples = []

    _example_cache[cache_key] = examples
    return examples


def prefetch_all_examples(records: list[dict], k: int) -> None:
    """Pre-compute RAG examples for every unique (persona_id, image) pair.

    Runs synchronously before the async loop so all CLIP encoding stays
    single-threaded and the cache is warm when process_row runs.
    """
    unique = {(int(r["persona_id"]), r["image"]) for r in records}
    for pid, img_rel in tqdm_sync(unique, desc="RAG prefetch", unit="image"):
        fetch_examples(pid, img_rel, k)


def build_messages(persona: str, image_b64: str, examples: list[dict]) -> list:
    system_content = (
        "You are roleplaying as the following person. Rate the visual aesthetics of website "
        "screenshots exactly as this person would, based on their background, values, and "
        "worldview — not as an AI assistant.\n\n"
        + persona
    )

    user_content = []

    if examples:
        user_content.append({
            "type": "text",
            "text": "Here are some examples of websites you have rated before:\n",
        })
        for ex in examples:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ex['image_b64']}"},
            })
            try:
                rating_str = f"{float(ex['mean_response']):.1f}"
            except (ValueError, TypeError):
                rating_str = str(ex["mean_response"])
            user_content.append({
                "type": "text",
                "text": f"Your rating: {rating_str}\n",
            })
        user_content.append({
            "type": "text",
            "text": "Now rate the following website in the same way:\n",
        })
    else:
        user_content.append({
            "type": "text",
            "text": "Look at this website screenshot. Rate how visually beautiful it is to you "
                    "on a scale from 1 (not beautiful at all) to 9 (extremely beautiful).\n\n",
        })

    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
    })
    user_content.append({
        "type": "text",
        "text": (
            "Rate how visually beautiful this website is to you "
            "on a scale from 1 (not beautiful at all) to 9 (extremely beautiful).\n\n"
            "Respond using this format:\n"
            "Rating: <number from 1 to 9>\n\n"
            "Do not include any explanation or extra text."
        ),
    })

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def parse_rating(raw: str) -> float | None:
    match = re.search(r"rating[:\s]+([0-9]+(?:\.[0-9]+)?)", raw, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        return value if 1.0 <= value <= 9.0 else None
    return None


async def get_rating(sem: asyncio.Semaphore, persona: str, image_b64: str, examples: list[dict]) -> str | None:
    messages = build_messages(persona, image_b64, examples)
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=16,
                    temperature=0.0,
                    timeout=15.0,
                )
                return response.choices[0].message.content.strip()
            except (openai.RateLimitError, openai.APITimeoutError):
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(RETRY_BASE_SECS * attempt + random.uniform(0, 1))
    return None


async def process_row(sem: asyncio.Semaphore, row: dict, k: int) -> float | None:
    image_b64 = await asyncio.to_thread(load_image_b64, row["image"])
    # Cache is already warm from prefetch — this is a dict lookup.
    examples  = fetch_examples(int(row["persona_id"]), row["image"], k)
    for _ in range(MAX_RETRIES):
        raw    = await get_rating(sem, row["persona_description"], image_b64, examples)
        rating = parse_rating(raw) if raw else None
        if rating is not None:
            break
    return rating


async def main(limit: int | None, k: int = K_EXAMPLES):
    df = pd.read_csv(DATASET_PATH)
    if limit is not None:
        df = df.head(limit)

    records = df.to_dict("records")
    prefetch_all_examples(records, k)

    sem     = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks   = [process_row(sem, row, k) for row in records]
    results = await tqdm.gather(*tasks, desc="Rating websites", unit="record")

    pd.DataFrame({"model_response": results}).to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. {len(results):,} records saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N records (e.g. --limit 100). Omit for all records."
    )
    parser.add_argument(
        "--k", type=int, default=K_EXAMPLES, metavar="K",
        help=f"Number of in-context examples per query (default: {K_EXAMPLES}).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit, k=args.k))
