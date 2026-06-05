import argparse
import asyncio
import base64
import os
import random
import re
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
DATASET_PATH = "website_likability/test_dataset.csv"
IMAGE_BASE   = "website_likability/website-aesthetics-datasets-master/rating-based-dataset/preprocess/resized"
OUTPUT_PATH  = "website_likability/output_baseline.csv"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


_image_cache: dict[str, str] = {}

def load_image_b64(image_rel: str) -> str:
    if image_rel not in _image_cache:
        path = os.path.join(IMAGE_BASE, image_rel.lstrip("/"))
        with open(path, "rb") as f:
            _image_cache[image_rel] = base64.b64encode(f.read()).decode("utf-8")
    return _image_cache[image_rel]


def build_messages(persona: str, image_b64: str) -> list:
    system_content = (
        "You are roleplaying as the following person. Rate the visual aesthetics of website "
        "screenshots exactly as this person would, based on their background, values, and "
        "worldview — not as an AI assistant.\n\n"
        + persona
    )
    user_content = [
        {
            "type": "text",
            "text": (
                "Look at this website screenshot. Rate how visually beautiful it is to you "
                "on a scale from 1 (not beautiful at all) to 9 (extremely beautiful).\n\n"
                "Respond using this format:\n"
                "Rating: <number from 1 to 9>\n\n"
                "Do not include any explanation or extra text."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        },
    ]
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


async def get_rating(sem: asyncio.Semaphore, persona: str, image_b64: str) -> str | None:
    messages = build_messages(persona, image_b64)
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


async def process_row(sem: asyncio.Semaphore, row: dict) -> float | None:
    image_b64 = await asyncio.to_thread(load_image_b64, row["image"])
    for _ in range(MAX_RETRIES):
        raw    = await get_rating(sem, row["persona_description"], image_b64)
        rating = parse_rating(raw) if raw else None
        if rating is not None:
            break
    return rating


async def main(limit: int | None):
    df = pd.read_csv(DATASET_PATH)
    if limit is not None:
        df = df.head(limit)

    sem     = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks   = [process_row(sem, row) for row in df.to_dict("records")]
    results = await tqdm.gather(*tasks, desc="Rating websites", unit="record")

    pd.DataFrame({"model_response": results}).to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. {len(results):,} records saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N records (e.g. --limit 100). Omit for all records."
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit))
