import argparse
import asyncio
import csv
import json
import os
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

MAX_CONCURRENCY = 8
MAX_RETRIES     = 5
RETRY_BASE_SECS = 2

load_dotenv()

MODEL        = "qwen/qwen3-vl-8b-instruct"
DATASET_PATH = os.path.join("opinion_qa", "test_dataset.csv")
OUTPUT_PATH  = os.path.join("opinion_qa", "output_baseline.csv")

client = AsyncOpenAI(
    base_url=os.getenv("OPENROUTER_URL"),
    api_key=os.getenv("OPENROUTER_KEY"),
)


def build_prompt(query: str, options: list[str]) -> str:
    opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
    return (
        f"Question: {query}\n\nOptions:\n{opts}\n\n"
        "Respond using this format:\n"
        "Answer: <the exact text of your chosen option>\n\n"
        "Do not include any explanation or extra text."
    )


def parse_choice(raw: str, options: list[str]) -> str | None:
    text = raw.strip().removeprefix("Answer:").removeprefix("answer:").strip()
    text_lower = text.lower()
    for opt in options:
        if opt.lower() in text_lower or text_lower in opt.lower():
            return opt
    return None


async def call_api(sem: asyncio.Semaphore, messages: list[dict]) -> str:
    """Call the API with per-attempt semaphore acquisition and exponential backoff."""
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


async def process_record(sem: asyncio.Semaphore, record: dict) -> str | None:
    options  = json.loads(record["options"])
    messages = []
    if persona := record.get("persona_description"):
        messages.append({"role": "system", "content": (
            "You are roleplaying as the following person. Answer every question exactly as they would "
            "based on their background, values, and worldview — not as an AI assistant.\n\n" + persona
        )})
    messages.append({"role": "user", "content": build_prompt(record["query"], options)})

    raw = await call_api(sem, messages)
    return parse_choice(raw, options) if raw else None


async def main(limit: int | None = None):
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if limit is not None:
        records = records[:limit]

    sem     = asyncio.Semaphore(MAX_CONCURRENCY)
    results = await tqdm.gather(
        *[process_record(sem, rec) for rec in records],
        desc="Processing", unit="record",
    )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_response"])
        writer.writerows([[r] for r in results])

    print(f"\nDone. {len(results):,} records saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline LLM on opinion-QA test set.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N records (default: all).",
    )
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))
