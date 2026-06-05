import argparse
import asyncio
import csv
import json
import os
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from build_vector_db_opinion_qa import VECTOR_DB_PATH, EMBEDDING_MODEL

MAX_CONCURRENCY      = 8
MAX_RETRIES          = 5
RETRY_BASE_SECS      = 2
K_EXAMPLES           = 3
MIN_EXAMPLE_DISTANCE = 0.32

load_dotenv()

MODEL        = "qwen/qwen3-vl-8b-instruct"
DATASET_PATH = os.path.join("opinion_qa", "test_dataset.csv")
OUTPUT_PATH  = os.path.join("opinion_qa", "output_model.csv")

client = AsyncOpenAI(
    base_url=os.getenv("OPENROUTER_URL"),
    api_key=os.getenv("OPENROUTER_KEY"),
)

# Initialised once — loading the embedding model is expensive.
_chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
_ef            = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# Cache keyed by (persona_id, question_key, k) — same question in same persona
# always returns the same examples, so no need to re-embed.
_example_cache: dict[tuple, list[dict]] = {}


def fetch_examples(persona_id: int, query: str, question_key: str, k: int) -> list[dict]:
    """Return top-k (query, human_response) pairs from the vector store.

    Results are cached by (persona_id, question_key, k) to avoid re-embedding
    the same question when it appears multiple times in the dataset.

    Preference order:
    1. distance >= MIN_EXAMPLE_DISTANCE  (genuinely different question)
    2. closer distance                   (fallback to always meet the k quota)
    Same question_key is always excluded via ChromaDB where-filter.
    """
    cache_key = (persona_id, question_key, k)
    if cache_key in _example_cache:
        return _example_cache[cache_key]

    try:
        collection = _chroma_client.get_collection(
            name=f"persona_{persona_id}", embedding_function=_ef,
        )
        results = collection.query(
            query_texts=[query],
            n_results=max(k * 5, 20),
            where={"question_key": {"$ne": question_key}},
        )
        candidates = [
            {"query": doc, "human_response": meta["human_response"], "_dist": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        preferred = [c for c in candidates if c["_dist"] >= MIN_EXAMPLE_DISTANCE]
        fallback  = [c for c in candidates if c["_dist"] <  MIN_EXAMPLE_DISTANCE]
        selected  = (preferred + fallback)[:k]
        examples  = [{"query": c["query"], "human_response": c["human_response"]} for c in selected]
    except Exception as e:
        print(f"  [fetch_examples] persona={persona_id} key={question_key}: {e}")
        examples = []

    _example_cache[cache_key] = examples
    return examples


def prefetch_all_examples(records: list[dict], k: int) -> None:
    """Pre-compute RAG examples for every unique (persona_id, question_key) pair.

    Running this synchronously before the async loop keeps all embedding work
    on a single thread, avoids asyncio.to_thread overhead, and populates the
    cache so process_record never blocks on DB lookups.
    """
    unique = {(int(r["persona_id"]), r["question_key"]) for r in records}
    # Build a lookup so we have the query text for each unique pair
    pair_to_query = {
        (int(r["persona_id"]), r["question_key"]): r["query"]
        for r in records
    }
    for pid, qkey in tqdm_sync(unique, desc="RAG prefetch", unit="question"):
        fetch_examples(pid, pair_to_query[(pid, qkey)], qkey, k)


def build_prompt(query: str, options: list[str], examples: list[dict]) -> str:
    opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
    parts = []
    if examples:
        parts.append("Here are some example questions and how a person like you answered them:\n")
        for ex in examples:
            parts.append(f"Q: {ex['query']}")
            parts.append(f"A: {ex['human_response']}\n")
        parts.append("Now answer the following question in the same way:\n")
    parts.append(
        f"Question: {query}\n\nOptions:\n{opts}\n\n"
        "Respond using this format:\n"
        "Answer: <the exact text of your chosen option>\n\n"
        "Do not include any explanation or extra text."
    )
    return "\n".join(parts)


def parse_choice(raw: str, options: list[str]) -> str | None:
    text = raw.strip().removeprefix("Answer:").removeprefix("answer:").strip()
    text_lower = text.lower()
    for opt in options:
        if opt.lower() in text_lower or text_lower in opt.lower():
            return opt
    return None


async def call_api(sem: asyncio.Semaphore, messages: list[dict]) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=32,
                    temperature=0.0,
                )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(RETRY_BASE_SECS ** attempt)


async def process_record(sem: asyncio.Semaphore, record: dict, k: int) -> str | None:
    options  = json.loads(record["options"])
    # Cache is already warm — this is a dict lookup, not a DB call.
    examples = fetch_examples(int(record["persona_id"]), record["query"], record["question_key"], k)

    messages = []
    if persona := record.get("persona_description"):
        messages.append({"role": "system", "content": (
            "You are roleplaying as the following person. Answer every question exactly as they would "
            "based on their background, values, and worldview — not as an AI assistant.\n\n" + persona
        )})
    messages.append({"role": "user", "content": build_prompt(record["query"], options, examples)})

    raw = await call_api(sem, messages)
    return parse_choice(raw, options) if raw else None


async def main(num_records: int | None = None, k: int = K_EXAMPLES):
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if num_records is not None:
        records = records[:num_records]

    prefetch_all_examples(records, k)

    sem     = asyncio.Semaphore(MAX_CONCURRENCY)
    results = await tqdm.gather(
        *[process_record(sem, rec, k) for rec in records],
        desc="LLM inference", unit="record",
    )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_response"])
        writer.writerows([[r] for r in results])

    print(f"\nDone. {len(results):,} records saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG-augmented LLM on opinion-QA test set."
    )
    parser.add_argument(
        "--num-records", type=int, default=None, metavar="N",
        help="Process only the first N records (default: all).",
    )
    parser.add_argument(
        "--k", type=int, default=K_EXAMPLES, metavar="K",
        help=f"Number of in-context examples per query (default: {K_EXAMPLES}).",
    )
    args = parser.parse_args()
    asyncio.run(main(num_records=args.num_records, k=args.k))
