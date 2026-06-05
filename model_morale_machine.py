import argparse
import asyncio
import os
import random
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from build_vector_db_morale_machine import VECTOR_DB_PATH, EMBEDDING_MODEL

MAX_CONCURRENCY      = 32
MAX_RETRIES          = 5
RETRY_BASE_SECS      = 2
K_EXAMPLES           = 3
MIN_EXAMPLE_DISTANCE = 0.32

load_dotenv()

API_KEY  = os.getenv("OPENROUTER_KEY")
BASE_URL = os.getenv("OPENROUTER_URL")

MODEL        = "qwen/qwen3-vl-8b-instruct"
DATASET_PATH = "morale_machine/test_dataset.csv"
OUTPUT_PATH  = "morale_machine/output_model.csv"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Initialised once — loading the embedding model is expensive.
_chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
_ef            = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

_example_cache: dict[tuple, list[dict]] = {}


def fetch_examples(persona_id: int, query: str, scenario_id: str, k: int) -> list[dict]:
    """Return top-k similar (scenario, human_choice) pairs from the vector store.

    Preference order:
    1. distance >= MIN_EXAMPLE_DISTANCE  (genuinely different scenario)
    2. closer distance                   (fallback to always meet the k quota)
    Same scenario_id is always excluded via ChromaDB where-filter.
    """
    cache_key = (persona_id, scenario_id, k)
    if cache_key in _example_cache:
        return _example_cache[cache_key]

    try:
        collection = _chroma_client.get_collection(
            name=f"persona_{persona_id}", embedding_function=_ef,
        )
        n_request = min(max(k * 5, 20), collection.count())
        results = collection.query(
            query_texts=[query],
            n_results=n_request,
            where={"scenario_id": {"$ne": str(scenario_id)}},
        )
        candidates = [
            {"scenario": doc, "human_choice": meta["human_choice"], "_dist": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        preferred = [c for c in candidates if c["_dist"] >= MIN_EXAMPLE_DISTANCE]
        fallback  = [c for c in candidates if c["_dist"] <  MIN_EXAMPLE_DISTANCE]
        selected  = (preferred + fallback)[:k]
        examples  = [{"scenario": c["scenario"], "human_choice": c["human_choice"]} for c in selected]
    except Exception as e:
        print(f"  [fetch_examples] persona={persona_id} scenario_id={scenario_id}: {e}")
        examples = []

    _example_cache[cache_key] = examples
    return examples


def prefetch_all_examples(records: list[dict], k: int) -> None:
    """Pre-compute RAG examples for every unique (persona_id, scenario_id) pair.

    Running this synchronously before the async loop keeps all embedding work
    on a single thread, avoids asyncio.to_thread overhead, and populates the
    cache so process_row never blocks on DB lookups.
    """
    unique = {(int(r["persona_id"]), str(r["scenario_id"])) for r in records}
    pair_to_query = {
        (int(r["persona_id"]), str(r["scenario_id"])): r["scenario"]
        for r in records
    }
    for pid, sid in tqdm_sync(unique, desc="RAG prefetch", unit="scenario"):
        fetch_examples(pid, pair_to_query[(pid, sid)], sid, k)


def _scenario_preamble(scenario: str) -> str:
    """Strip response-format instructions from a scenario string for use in examples."""
    idx = scenario.find("Respond using this format:")
    return scenario[:idx].rstrip() if idx != -1 else scenario


def build_messages(scenario: str, persona: str, examples: list[dict]) -> list:
    parts = []
    if examples:
        parts.append("Here are some examples of moral dilemmas and how a person like you decided:\n")
        for ex in examples:
            parts.append(_scenario_preamble(ex["scenario"]))
            parts.append(f"Decision: {ex['human_choice']}\n")
        parts.append("Now decide the following moral dilemma in the same way:\n")
    parts.append(scenario)

    return [
        {
            "role": "system",
            "content": (
                "You are roleplaying as the following person. Make every moral decision "
                "exactly as they would, based on their background, values, and worldview "
                "— not as an AI assistant.\n\n"
                + persona
            ),
        },
        {
            "role": "user",
            "content": "\n".join(parts),
        },
    ]


def parse_decision(raw: str) -> str | None:
    """Extract 'Option A' or 'Option B' from the model response."""
    text = raw.strip()
    if text.lower().startswith("decision:"):
        text = text[len("decision:"):].strip()
    text_lower = text.lower()
    if "option a" in text_lower:
        return "Option A"
    if "option b" in text_lower:
        return "Option B"
    return None


async def get_response(sem: asyncio.Semaphore, scenario: str, persona: str, examples: list[dict]) -> str | None:
    messages = build_messages(scenario, persona, examples)
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


async def process_row(sem: asyncio.Semaphore, row: dict, k: int) -> str | None:
    # Cache is already warm from prefetch — this is a dict lookup.
    examples = fetch_examples(int(row["persona_id"]), row["scenario"], str(row["scenario_id"]), k)
    for _ in range(MAX_RETRIES):
        raw      = await get_response(sem, row["scenario"], row["persona_description"], examples)
        decision = parse_decision(raw) if raw else None
        if decision is not None:
            break
    return decision


async def main(limit: int | None, k: int = K_EXAMPLES) -> None:
    df = pd.read_csv(DATASET_PATH)
    if limit is not None:
        df = df.head(limit)

    records = df.to_dict("records")
    prefetch_all_examples(records, k)

    sem     = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks   = [process_row(sem, row, k) for row in records]
    results = await tqdm.gather(*tasks, desc="Processing", unit="scenario")

    pd.DataFrame({"model_response": results}).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nDone. {len(results):,} results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N records (e.g. --limit 100). Omit for all.",
    )
    parser.add_argument(
        "--k", type=int, default=K_EXAMPLES, metavar="K",
        help=f"Number of in-context examples per query (default: {K_EXAMPLES}).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit, k=args.k))
