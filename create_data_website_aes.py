import argparse
import pandas as pd
from persona_helper import load_personas

BASE        = "website_likability/website-aesthetics-datasets-master/rating-based-dataset"
CSV_PATH    = f"{BASE}/data/ae_only_unambiguous_1000.csv"
TEST_PATH   = f"{BASE}/preprocess/test_list.csv"
OUTPUT_DIR  = "website_likability"

parser = argparse.ArgumentParser(description="Build website aesthetics dataset with persona assignments.")
parser.add_argument("--num-rows", type=int, default=1000, metavar="N",
                    help="Limit output to the first N rows (default: all).")
args = parser.parse_args()


# ── 1. Load main CSV and add image path ──────────────────────────────────────
print("Loading main CSV…")
df = pd.read_csv(CSV_PATH, low_memory=False)
df["image"] = df["website"].apply(
    lambda x: "/{}_resized/{}.png".format(*x.rsplit("_", 1))
)

# ── 2. Load test list and keep only test images ───────────────────────────────
test_images = set(pd.read_csv(TEST_PATH)["image"])
df = df[df["image"].isin(test_images)]
df = df.rename_axis("row_id").reset_index()

# ── 3. Build persona lookup dict keyed by (gender, age_bucket) ───────────────
personas = load_personas()

persona_dict: dict[tuple, list[tuple]] = {}
for p in personas:
    wf = p["webaes_loose_filters"]
    persona_dict.setdefault((int(wf["gender.x"]), wf["age_bucket"]), []).append(
        (p["persona_id"], p["description"])
    )

# ── 4. Derive lookup keys — drop gender=2, vectorised age bucketing ───────────
df = df[df["gender.x"].astype(int) != 2].copy()
df["gender_key"] = df["gender.x"].astype(int)
df["age_bucket"] = pd.cut(
    df["age.x"],
    bins=[0, 24, 34, 44, 54, float("inf")],
    labels=["18-24", "25-34", "35-44", "45-54", "55+"],
)

# ── 5. Expand each row into k rows — one per matching persona ─────────────────
print("Assigning personas…")
df["persona_key"] = list(zip(df["gender_key"], df["age_bucket"]))
df["_personas"]   = df["persona_key"].map(persona_dict)
df = df[df["_personas"].notna()]
df = df.explode("_personas").reset_index(drop=True)
df["persona_id"]          = df["_personas"].apply(lambda x: x[0])
df["persona_description"] = df["_personas"].apply(lambda x: x[1])
df = df.drop(columns=["_personas"])

# ── 6. Build final dataset ────────────────────────────────────────────────────
final_df = (
    df[["row_id", "image", "age.x", "gender.x", "persona_key",
        "persona_id", "persona_description", "mean_response", "std_response", "difference_response"]]
    .rename(columns={"age.x": "age", "gender.x": "gender"})
)

if args.num_rows is not None:
    final_df = pd.concat(
        [g.sample(min(len(g), args.num_rows), random_state=42)
         for _, g in final_df.groupby("persona_id")]
    ).reset_index(drop=True)

# ── 7. Show persona IDs with no data ─────────────────────────────────────────
all_ids     = {p["persona_id"] for p in personas}
covered_ids = set(final_df["persona_id"])
missing_ids = sorted(all_ids - covered_ids)
if missing_ids:
    print(f"Persona IDs with no data : {missing_ids}")
else:
    print("All persona IDs have data.")

# ── 8. Shuffle and split 50/50, save ─────────────────────────────────────────
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
split    = len(final_df) // 2
splits   = {"train": final_df.iloc[:split], "test": final_df.iloc[split:]}

for name, df_split in splits.items():
    path = f"{OUTPUT_DIR}/{name}_dataset.csv"
    df_split.to_csv(path, index=False)
    print(f"Saved {len(df_split):,} rows -> {path}")
