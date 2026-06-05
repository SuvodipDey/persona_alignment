import os
import sys
import numpy as np
import pandas as pd

TEST_DATASET_PATH = "website_likability/test_dataset.csv"
BASELINE_PATH     = "website_likability/output_baseline.csv"
MODEL_PATH        = "website_likability/output_model.csv"
OUTPUT_PATH       = "website_likability/website_aes_result.csv"
MIN_WEBSITES      = 2


def compute_pearson_r(group: pd.DataFrame, rating_col: str) -> tuple[float, int]:
    """
    Compute Pearson r between human mean_response and model rating for a persona group.

    Deduplicates by image first (averages model ratings per unique image), then
    correlates with mean_response.

    Returns (pearson_r, n_websites).
    """
    per_image = (
        group.dropna(subset=[rating_col, "mean_response"])
        .groupby("image")
        .agg(mean_response=("mean_response", "first"),
             model_rating=(rating_col,        "mean"))
        .reset_index()
    )
    if len(per_image) < MIN_WEBSITES:
        return float("nan"), len(per_image)
    return round(per_image["mean_response"].corr(per_image["model_rating"]), 6), len(per_image)


def main() -> None:
    # ── 1. Check input files exist ────────────────────────────────────────────
    missing = [
        (BASELINE_PATH, "python run_baseline_website_aes.py"),
        (MODEL_PATH,    "python model_website_aes.py"),
    ]
    missing = [(p, cmd) for p, cmd in missing if not os.path.exists(p)]
    if missing:
        print("Required output file(s) not found. Generate them first:\n")
        for path, cmd in missing:
            print(f"  Missing : {path}")
            print(f"  Run     : {cmd}\n")
        sys.exit(1)

    # ── 2. Load data, align row counts ───────────────────────────────────────
    baseline_responses = pd.read_csv(BASELINE_PATH)["model_response"]
    model_responses    = pd.read_csv(MODEL_PATH)["model_response"]
    test_df            = pd.read_csv(TEST_DATASET_PATH)

    n_use = min(len(baseline_responses), len(model_responses), len(test_df))
    # if n_use < len(test_df):
    #     print(
    #         f"Warning: output files have fewer rows than test dataset "
    #         f"(baseline={len(baseline_responses)}, model={len(model_responses)}, "
    #         f"test={len(test_df)}). Using first {n_use} rows.\n"
    #     )

    test_df = test_df.iloc[:n_use].reset_index(drop=True)
    test_df["rating_baseline"] = baseline_responses.iloc[:n_use].values
    test_df["rating_model"]    = model_responses.iloc[:n_use].values

    n_parsed_b = test_df["rating_baseline"].notna().sum()
    n_parsed_m = test_df["rating_model"].notna().sum()
    # print(f"Records used       : {n_use:,}")
    # print(f"Parsed (baseline)  : {n_parsed_b:,} ({n_parsed_b / n_use * 100:.1f}%)")
    # print(f"Parsed (model)     : {n_parsed_m:,} ({n_parsed_m / n_use * 100:.1f}%)\n")

    # ── 3. Per-persona Pearson r ──────────────────────────────────────────────
    summary_rows = []
    for persona_id, group in test_df.groupby("persona_id"):
        persona_desc = group["persona_description"].iloc[0]

        r_b, n_sites_b = compute_pearson_r(group, "rating_baseline")
        r_m, n_sites_m = compute_pearson_r(group, "rating_model")

        pct = round((r_m - r_b) * 100 / r_b, 2) if r_b and not np.isnan(r_b) else float("nan")

        summary_rows.append({
            "persona_id":          persona_id,
            "persona_description": persona_desc,
            "n_records":           len(group),
            "n_websites":          n_sites_b,
            "pearson_r_baseline":  r_b,
            "pearson_r_model":     r_m,
            "pct_improvement":     pct,
        })

        # print("=" * 72)
        # print(f"Persona ID  : {persona_id}")
        # print(f"Persona     : {persona_desc}")
        # print(f"Records     : {len(group)}   |   Websites : {n_sites_b}/{n_sites_m}")
        # print(f"Pearson r   : baseline={r_b}  model={r_m}  pct_improvement={pct if not np.isnan(pct) else 'n/a'}")
        # print()

    if not summary_rows:
        print("No personas found with sufficient data.")
        return

    # ── 4. Build result table + Overall row ───────────────────────────────────
    result_df = pd.DataFrame(summary_rows).sort_values("persona_id").reset_index(drop=True)

    mean_r_b    = result_df["pearson_r_baseline"].mean()
    mean_r_m    = result_df["pearson_r_model"].mean()
    overall_pct = round((mean_r_m - mean_r_b) * 100 / mean_r_b, 2) if mean_r_b and not np.isnan(mean_r_b) else float("nan")

    result_df = pd.concat([result_df, pd.DataFrame([{
        "persona_id":          "Overall",
        "persona_description": "",
        "n_records":           result_df["n_records"].sum(),
        "n_websites":          result_df["n_websites"].sum(),
        "pearson_r_baseline":  round(mean_r_b, 6),
        "pearson_r_model":     round(mean_r_m, 6),
        "pct_improvement":     overall_pct,
    }])], ignore_index=True)

    print("=" * 72)
    print("SUMMARY — ordered by persona_id")
    print("=" * 72)
    print(result_df.to_string(index=False))
    print()
    print(f"Mean Pearson r (baseline) : {mean_r_b:.6f}")
    print(f"Mean Pearson r (model)    : {mean_r_m:.6f}")
    print(f"Overall % improvement     : {overall_pct:.2f}%")

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
