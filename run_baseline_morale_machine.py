import argparse
import asyncio
import os
import random
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import pandas as pd

MAX_CONCURRENCY = 32
MAX_RETRIES     = 5
RETRY_BASE_SECS = 2

load_dotenv()

API_KEY  = os.getenv("OPENROUTER_KEY")
BASE_URL = os.getenv("OPENROUTER_URL")

MODEL        = "qwen/qwen3-vl-8b-instruct"
DATASET_PATH = "morale_machine/test_dataset.csv"
OUTPUT_PATH  = "morale_machine/output_baseline.csv"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


def build_messages(scenario: str, persona: str) -> list:
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
            "content": scenario,
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


async def get_response(sem: asyncio.Semaphore, scenario: str, persona: str) -> str | None:
    messages = build_messages(scenario, persona)
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


async def process_row(sem: asyncio.Semaphore, row: dict) -> str | None:
    for _ in range(MAX_RETRIES):
        raw      = await get_response(sem, row["scenario"], row["persona_description"])
        decision = parse_decision(raw) if raw else None
        if decision is not None:
            break
    return decision


async def main(limit: int | None) -> None:
    df = pd.read_csv(DATASET_PATH)
    if limit is not None:
        df = df.head(limit)

    sem     = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks   = [process_row(sem, row) for row in df.to_dict("records")]
    results = await tqdm.gather(*tasks, desc="Processing", unit="scenario")

    pd.DataFrame({"model_response": results}).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nDone. {len(results):,} results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N records (e.g. --limit 100). Omit for all.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit))
