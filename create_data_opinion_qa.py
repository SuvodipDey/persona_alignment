import argparse
import ast
import csv
import json
import os
import random
from collections import defaultdict

from persona_helper import load_personas, find_best_persona, PERSONAS_FILE

HUMAN_RESP_DIR = "./opinion_qa/data/human_resp"
DISAGREEMENT_SURVEY = "Pew_American_Trends_Panel_disagreement_500"
OUTPUT_DIR = "./opinion_qa"
MIN_MATCHING_FILTERS = 8

DEMOGRAPHIC_ATTRIBUTES = [
    "CREGION", "AGE", "SEX", "EDUCATION",
    "MARITAL", "RELIG", "POLPARTY",
    "INCOME", "POLIDEOLOGY", "RACE",
]

ATTRIBUTE_VALUE_MAP = {
    "RACE": {"Mixed Race": "Other"},
    "RELIG": {"Christian": "Protestant", "Unitarian": "Protestant"},
}

CSV_COLUMNS = [
    "survey", "question_key", "query", "options", "option_ordinal",
    "human_response", "attributes", "persona_id", "persona_description",
    "persona_matching", "unmatched_attributes",
]


def extract_attributes(respondent):
    attrs = {}
    for attr in DEMOGRAPHIC_ATTRIBUTES:
        val = respondent.get(attr, "").strip()
        if not val or val == "Refused":
            continue
        attrs[attr] = ATTRIBUTE_VALUE_MAP.get(attr, {}).get(val, val)
    return attrs


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_question_row(row):
    """Return (key, query, options, ordinals) for one info.csv row, or None if no key."""
    key = row.get("key", "").strip()
    if not key:
        return None
    all_options = ast.literal_eval(row.get("references", "[]"))
    ordinals = ast.literal_eval(row.get("option_ordinal", "[]"))
    return key, row.get("question", "").strip(), [o for o in all_options if o != "Refused"], ordinals


def parse_questions(info_rows):
    questions = {}
    for row in info_rows:
        parsed = _parse_question_row(row)
        if parsed is None:
            continue
        key, query, options, ordinals = parsed
        questions[key] = {"query": query, "options": options, "option_ordinal": ordinals}
    return questions


def parse_disagreement_questions(info_rows):
    """Group disagreement_500 questions by source wave directory.

    The `survey` column contains names like 'Pew_American_Trends_Panel_W26';
    stripping 'Pew_' gives the human_resp directory name.
    Returns: {wave_dir: {qkey: {query, options, option_ordinal}}}
    """
    by_wave = {}
    for row in info_rows:
        parsed = _parse_question_row(row)
        if parsed is None:
            continue
        key, query, options, ordinals = parsed
        wave_dir = row.get("survey", "").strip().removeprefix("Pew_")
        by_wave.setdefault(wave_dir, {})[key] = {
            "query": query, "options": options, "option_ordinal": ordinals,
        }
    return by_wave


def process_survey(label, questions, responses_path, personas):
    """Match respondents to their best persona, print per-survey stats, return records."""
    print(f"Processing {label} ...", end=" ", flush=True)

    records = []
    matched = unmatched = 0

    for respondent in read_csv(responses_path):
        attributes = extract_attributes(respondent)
        persona_id, persona_description, n_passing, _, failing_filters = find_best_persona(
            attributes, personas,
            consider_keys=DEMOGRAPHIC_ATTRIBUTES,
            min_passing=MIN_MATCHING_FILTERS,
        )

        if persona_description:
            matched += 1
        else:
            unmatched += 1
            continue

        for qkey, qinfo in questions.items():
            answer = respondent.get(qkey, "").strip()
            if not answer or answer == "Refused":
                continue
            records.append({
                "survey": label,
                "question_key": qkey,
                "query": qinfo["query"],
                "options": json.dumps(qinfo["options"], ensure_ascii=False),
                "option_ordinal": json.dumps(qinfo["option_ordinal"], ensure_ascii=False),
                "human_response": answer,
                "attributes": json.dumps(attributes, ensure_ascii=False),
                "persona_id": persona_id,
                "persona_description": persona_description,
                "persona_matching": n_passing,
                "unmatched_attributes": json.dumps(failing_filters, ensure_ascii=False),
                "_n_passing": n_passing,
            })

    total = matched + unmatched
    print(
        f"{len(records):,} records | "
        f"respondents: {matched}/{total} matched, "
        f"{unmatched}/{total} skipped ({100 * unmatched / total:.1f}%)"
    )
    return records


def sample_per_persona(all_records, num_rows):
    """Keep num_rows randomly sampled records per persona_id."""
    by_persona = defaultdict(list)
    for rec in all_records:
        by_persona[rec["persona_id"]].append(rec)

    sampled = []
    for recs in by_persona.values():
        sampled.extend(random.sample(recs, min(num_rows, len(recs))))
    return sampled


def build_dataset(include_disagreement=False, num_rows=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    personas = load_personas(PERSONAS_FILE)

    survey_dirs = sorted(
        d for d in os.listdir(HUMAN_RESP_DIR)
        if os.path.isdir(os.path.join(HUMAN_RESP_DIR, d))
        and os.path.exists(os.path.join(HUMAN_RESP_DIR, d, "responses.csv"))
    )

    all_records = []

    for survey_name in survey_dirs:
        survey_path = os.path.join(HUMAN_RESP_DIR, survey_name)
        questions = parse_questions(read_csv(os.path.join(survey_path, "info.csv")))
        all_records.extend(process_survey(
            survey_name, questions,
            os.path.join(survey_path, "responses.csv"),
            personas,
        ))

    if include_disagreement:
        disagreement_dir = os.path.join(HUMAN_RESP_DIR, DISAGREEMENT_SURVEY)
        for wave_dir, questions in sorted(parse_disagreement_questions(
            read_csv(os.path.join(disagreement_dir, "info.csv"))
        ).items()):
            responses_path = os.path.join(HUMAN_RESP_DIR, wave_dir, "responses.csv")
            if not os.path.exists(responses_path):
                print(f"  [{DISAGREEMENT_SURVEY}] Skipping {wave_dir} — no responses.csv")
                continue
            all_records.extend(process_survey(
                f"{DISAGREEMENT_SURVEY} / {wave_dir}", questions, responses_path, personas,
            ))

    if num_rows is not None:
        before = len(all_records)
        all_records = sample_per_persona(all_records, num_rows)
        print(f"\nSampling: {before:,} → {len(all_records):,} records "
              f"(max {num_rows:,} per persona, random)")

    random.shuffle(all_records)
    split = len(all_records) // 2
    splits = {"train": all_records[:split], "test": all_records[split:]}

    records_per_persona = defaultdict(int)
    for label, records in splits.items():
        out_file = os.path.join(OUTPUT_DIR, f"{label}_dataset.csv")
        with open(out_file, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
                records_per_persona[rec["persona_id"]] += 1
        print(f"\n{label.capitalize()} set: {len(records):,} records saved -> {out_file}")

    print(f"\nTotal: {len(all_records):,} records across both splits")
    #print("\nRecords per persona (combined):")
    #for pid, count in sorted(records_per_persona.items()):
    #    print(f"  Persona {pid:2d}: {count:,} records")

    empty_ids = sorted({p["persona_id"] for p in personas} - set(records_per_persona))
    if empty_ids:
        print(f"\nPersonas with no records: {empty_ids}")
    else:
        print("\nAll personas have at least one record.")


def main():
    parser = argparse.ArgumentParser(
        description="Build opinion-QA dataset from Pew survey responses."
    )
    parser.add_argument(
        "--include-disagreement",
        action="store_true",
        default=False,
        help="Also process Pew_American_Trends_Panel_disagreement_500 (default: excluded).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=1000,
        metavar="N",
        help="Number of randomly sampled records per persona_id (default: 1000).",
    )
    args = parser.parse_args()
    build_dataset(
        include_disagreement=args.include_disagreement,
        num_rows=args.num_rows,
    )


if __name__ == "__main__":
    main()
