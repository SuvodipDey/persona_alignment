import argparse
import json
import os
import tarfile
import pandas as pd

BASE_DIR      = "morale_machine/Datasets/Moral Machine Data"
CSV_PATH      = f"{BASE_DIR}/SharedResponsesSurvey.csv"
CSV_GZ_PATH   = f"{BASE_DIR}/SharedResponsesSurvey.csv.tar.gz"
PERSONAS_PATH = "personas_40_selected.json"

SCENARIO_COLS = [
    "ResponseID", "UserID", "ScenarioOrder",
    "Intervention", "PedPed", "Barrier", "CrossingSignal",
    "AttributeLevel", "ScenarioType", "ScenarioTypeStrict",
    "DefaultChoice", "NonDefaultChoice", "DefaultChoiceIsOmission",
    "NumberOfCharacters", "DiffNumberOFCharacters",
    "Saved",
]
DEMO_COLS = [
    "UserCountry3",
    "Review_age", "Review_gender", "Review_education",
    "Review_income", "Review_political", "Review_religious",
]

MATCH_ATTRS      = ("SEX", "AGE", "POLIDEOLOGY", "INCOME", "EDUCATION")
MIN_MATCH_ATTRS  = 3

_GENDER_MAP = {"male": "Male", "female": "Female"}

_INCOME_MAP = {
    "under5000":   "Less than $30,000",
    "5000":        "Less than $30,000",
    "10000":       "Less than $30,000",
    "15000":       "Less than $30,000",
    "25000":       "Less than $30,000",
    "35000":       "$30,000-$50,000",
    "50000":       "$50,000-$75,000",
    "80000":       "$75,000-$100,000",
    "above100000": "$100,000 or more",
}

_EDU_MAP = {
    "underHigh":  "Less than high school",
    "high":       "High school graduate",
    "vocational": "Some college, no degree",
    "college":    "Some college, no degree",
    "bachelor":   "College graduate/some postgrad",
    "graduate":   "Postgraduate",
}


# ── Demographic mapping helpers (used for raw columns in output) ──────────────

def age_to_webaes_bucket(age) -> str | None:
    try:
        age = float(age)
    except (ValueError, TypeError):
        return None
    if age <= 24:  return "18-24"
    if age <= 34:  return "25-34"
    if age <= 44:  return "35-44"
    if age <= 54:  return "45-54"
    return "55+"


def gender_to_key(gender) -> int | None:
    g = str(gender).strip().lower()
    if g == "male":   return 0
    if g == "female": return 1
    return None


# ── Vectorised feature extraction ─────────────────────────────────────────────

def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add _sex, _age, _income, _edu, _ideology columns to a stay-row DataFrame."""
    df = df.copy()

    df["_sex"] = df["Review_gender"].str.strip().str.lower().map(_GENDER_MAP)

    df["_age"] = pd.cut(
        pd.to_numeric(df["Review_age"], errors="coerce"),
        bins=[0, 29, 49, 64, float("inf")],
        labels=["18-29", "30-49", "50-64", "65+"],
    ).astype(object)

    df["_income"] = df["Review_income"].astype(str).str.strip().map(_INCOME_MAP)
    df["_edu"]    = df["Review_education"].astype(str).str.strip().map(_EDU_MAP)

    pol = pd.to_numeric(df["Review_political"], errors="coerce")
    ideology = pd.cut(
        pol,
        bins=[0, 0.30, 0.45, 0.55, 0.70, 1.0],
        labels=["Very conservative", "Conservative", "Moderate", "Liberal", "Very liberal"],
        include_lowest=True,   # include 0.0 → "Very conservative"
    ).astype(object)
    ideology[pol == 0.5] = None   # 0.5 = "no answer" sentinel
    df["_ideology"] = ideology

    return df


# ── Persona matching ─────────────────────────────────────────────────────────

def match_personas_partial(personas: list[dict], features: dict, min_match: int = MIN_MATCH_ATTRS) -> list[dict]:
    """Return all personas where at least min_match out of 5 attributes match.

    Only non-None annotator attributes are compared; None attributes are skipped.
    """
    result = []
    for p in personas:
        opa   = p["opa_filters"]
        count = sum(
            1 for attr in MATCH_ATTRS
            if features.get(attr) is not None and opa.get(attr) == features[attr]
        )
        if count >= min_match:
            result.append(p)
    return result


# ── Scenario text builders ────────────────────────────────────────────────────

_CHAR_NOUNS: dict[tuple, tuple[str, str]] = {
    ("Age",          "Young"):   ("young person",      "young people"),
    ("Age",          "Old"):     ("elderly person",    "elderly people"),
    ("Gender",       "Males"):   ("man",               "men"),
    ("Gender",       "Females"): ("woman",             "women"),
    ("Fitness",      "Fit"):     ("athletic person",   "athletic people"),
    ("Fitness",      "Fat"):     ("overweight person", "overweight people"),
    ("Social Value", "High"):    ("executive",         "executives"),
    ("Social Value", "Low"):     ("homeless person",   "homeless people"),
    ("Species",      "Hoomans"): ("person",            "people"),
    ("Species",      "Pets"):    ("animal",            "animals"),
}


def describe_group(row: pd.Series) -> str:
    n      = int(row["NumberOfCharacters"])
    stype  = str(row.get("ScenarioType", "")).strip()
    attr   = str(row.get("AttributeLevel", "")).strip()
    signal = int(row["CrossingSignal"]) if not pd.isna(row.get("CrossingSignal")) else 0

    singular, plural = _CHAR_NOUNS.get((stype, attr), ("person", "people"))
    noun = singular if n == 1 else plural

    if int(row["Barrier"]) == 1:
        context = f"riding as {'a passenger' if n == 1 else 'passengers'} in the car"
    elif signal == 1:
        context = "crossing the street on a green light"
    elif signal == 2:
        context = "crossing the street on a red light"
    else:
        context = "in the road"

    return f"{n} {noun} {context}"


def build_scenario_text(stay: pd.Series, swerve: pd.Series) -> tuple[str, str, str]:
    group_a = describe_group(stay)
    group_b = describe_group(swerve)

    if int(stay["PedPed"]) == 1:
        setup = (
            "A self-driving car with sudden brake failure must choose "
            "between two groups of pedestrians."
        )
    else:
        setup = (
            "A self-driving car with sudden brake failure must make an "
            "emergency decision between the safety of its passengers and "
            "pedestrians outside."
        )

    option_a = f"The car stays on course, killing {group_a}."
    option_b = f"The car swerves off course, killing {group_b}."
    scenario = (
        f"{setup}\n\n"
        f"Option A -- {option_a}\n"
        f"Option B -- {option_b}\n\n"
        f"Which option should the car choose?\n\n"
        f"Respond using this format:\n"
        f"Decision: <Option A or Option B>\n\n"
        f"Do not include any explanation or extra text."
    )
    return scenario, option_a, option_b


# ── CSV loading ───────────────────────────────────────────────────────────────

def resolve_csv() -> str:
    if os.path.exists(CSV_PATH):
        return CSV_PATH
    print("Extracting CSV from archive (one-time, ~1.8 GB)...")
    with tarfile.open(CSV_GZ_PATH, "r:gz") as tar:
        tar.extractall(path=BASE_DIR)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Extraction failed: {CSV_PATH} not found.")
    return CSV_PATH


def load_paired_sample(n_scenarios: int | None = None) -> pd.DataFrame:
    path      = resolve_csv()
    want_cols = set(SCENARIO_COLS + DEMO_COLS)

    print("  Reading CSV with selected columns...")
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in want_cols)
    print(f"  Total rows: {len(df):,} | Unique ResponseIDs: {df['ResponseID'].nunique():,}")

    counts   = df.groupby("ResponseID").size()
    complete = counts[counts == 2].index
    df       = df[df["ResponseID"].isin(complete)]
    print(f"  Complete pairs found: {len(complete):,}")

    if n_scenarios is not None and len(complete) > n_scenarios:
        sampled = pd.Series(complete).sample(n=n_scenarios, random_state=42).values
        df      = df[df["ResponseID"].isin(sampled)]
        print(f"  Sampled down to {n_scenarios:,} scenarios.")

    return df.reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_samples: int | None, n_scenarios: int | None) -> None:
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)

    for i, p in enumerate(personas, start=1):
        p["_id"] = i

    print(f"n_samples per persona = {n_samples or 'all'} | n_scenarios = {n_scenarios or 'all'} | min_match = {MIN_MATCH_ATTRS}/5")
    df = load_paired_sample(n_scenarios)

    # ── Pre-filter: keep stay rows with >= MIN_MATCH_ATTRS valid features ─────
    stay_df   = add_feature_columns(df[df["Intervention"] == 0])
    swerve_df = df[df["Intervention"] == 1].set_index("ResponseID")

    feat_cols  = ["_sex", "_age", "_income", "_edu", "_ideology"]
    total_stay = len(stay_df)
    non_null   = stay_df[feat_cols].notna().sum(axis=1)
    stay_df    = stay_df[non_null >= MIN_MATCH_ATTRS].set_index("ResponseID")

    valid_ids = stay_df.index.intersection(swerve_df.index)
    stay_df   = stay_df.loc[valid_ids]
    swerve_df = swerve_df.loc[valid_ids]
    print(f"  Stay rows with >= {MIN_MATCH_ATTRS} features valid : {len(valid_ids):,} / {total_stay:,} ({100*len(valid_ids)/total_stay:.1f}%)")

    # ── Build records — partial persona matching per scenario ─────────────────
    records = []
    for response_id, stay in stay_df.iterrows():
        swerve = swerve_df.loc[response_id]

        features = {
            "SEX":        stay["_sex"]      if pd.notna(stay["_sex"])      else None,
            "AGE":        stay["_age"]      if pd.notna(stay["_age"])      else None,
            "POLIDEOLOGY": stay["_ideology"] if pd.notna(stay["_ideology"]) else None,
            "INCOME":     stay["_income"]   if pd.notna(stay["_income"])   else None,
            "EDUCATION":  stay["_edu"]      if pd.notna(stay["_edu"])      else None,
        }
        matching = match_personas_partial(personas, features)
        if not matching:
            continue

        scenario, option_a, option_b = build_scenario_text(stay, swerve)
        human_choice = "Option A" if int(stay["Saved"]) == 0 else "Option B"

        for persona in matching:
            records.append({
                "scenario_id":         response_id,
                "ScenarioOrder":       stay["ScenarioOrder"],
                "ScenarioType":        stay["ScenarioType"],
                "ScenarioTypeStrict":  stay["ScenarioTypeStrict"],
                "scenario":            scenario,
                "option_a":            option_a,
                "option_b":            option_b,
                "human_choice":        human_choice,
                "UserCountry3":        stay["UserCountry3"],
                "Review_age":          stay["Review_age"],
                "Review_gender":       stay["Review_gender"],
                "Review_education":    stay["Review_education"],
                "Review_income":       stay["Review_income"],
                "Review_political":    stay["Review_political"],
                "Review_religious":    stay["Review_religious"],
                "annotator_sex":       features["SEX"],
                "annotator_age":       features["AGE"],
                "annotator_income":    features["INCOME"],
                "annotator_education": features["EDUCATION"],
                "annotator_ideology":  features["POLIDEOLOGY"],
                "persona_key":         (gender_to_key(stay["Review_gender"]),
                                        age_to_webaes_bucket(stay["Review_age"])),
                "persona_id":          persona["_id"],
                "persona_description": persona["description"],
            })

    final_df = pd.DataFrame(records)

    if n_samples is not None and not final_df.empty:
        final_df = pd.concat(
            [g.sample(min(len(g), n_samples), random_state=42)
             for _, g in final_df.groupby("persona_id")]
        ).reset_index(drop=True)

    os.makedirs("morale_machine", exist_ok=True)

    if final_df.empty:
        print("WARNING: no records saved. No annotators matched all 5 attributes.")
        return

    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = len(final_df) // 2
    splits   = {"train": final_df.iloc[:split], "test": final_df.iloc[split:]}

    for name, df_split in splits.items():
        path = os.path.join("morale_machine", f"{name}_dataset.csv")
        df_split.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df_split):,} records -> {path}")

    counts  = final_df["persona_id"].value_counts()
    all_ids = pd.RangeIndex(1, 41)
    counts  = counts.reindex(all_ids, fill_value=0).sort_index()

    missing = sorted(counts[counts == 0].index.tolist())
    covered = 40 - len(missing)

    print(f"\nPersonas covered : {covered}/40")
    print("\nRecords per persona:")
    print(counts.to_string())
    if missing:
        print(f"\nPersona IDs with no data: {missing}")
    else:
        print("\nAll persona IDs have data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Moral Machine persona dataset with exact 5-attribute matching."
    )
    parser.add_argument("--n_samples",   type=int, default=1000,
                        help="Max samples per persona_id in output (default: 1000).")
    parser.add_argument("--n_scenarios", type=int, default=100000,
                        help="Max scenarios to load from CSV (default: 100000).")
    args = parser.parse_args()
    main(args.n_samples, args.n_scenarios)
