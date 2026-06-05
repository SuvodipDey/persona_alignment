import json
import os
import sys
import numpy as np
import pandas as pd

TEST_DATASET_PATH = os.path.join("opinion_qa", "test_dataset.csv")
BASELINE_PATH     = os.path.join("opinion_qa", "output_baseline.csv")
MODEL_PATH        = os.path.join("opinion_qa", "output_model.csv")
OUTPUT_PATH       = os.path.join("opinion_qa", "opinion_qa_result.csv")


def total_variation_distance(p: dict, q: dict, options: list) -> float:
    return sum(abs(p.get(opt, 0.0) - q.get(opt, 0.0)) for opt in options) / 2.0


def empirical_dist(responses: list, options: list) -> dict:
    total  = len(responses)
    counts = {opt: 0 for opt in options}
    for r in responses:
        if r in counts:
            counts[r] += 1
    return {opt: counts[opt] / total for opt in options} if total else {opt: 0.0 for opt in options}


def compute_representativeness(group: pd.DataFrame, response_col: str) -> tuple[float, int]:
    """
    Rep(g) = 1 - (1/|Q|) * sum_q TVD( P^G(q) || P_hat^G(q) )

    P^G(q)     : empirical human response distribution for question q in group g
    P_hat^G(q) : empirical model response distribution for question q in group g
    |Q|        : number of unique questions with at least one parseable model response

    Returns (rep_score, n_questions).
    """
    valid = group[group[response_col].notna()].copy()

    tvds = []
    for _, q_df in valid.groupby("question_key"):
        options = q_df["options"].iloc[0]
        h_dist  = empirical_dist(q_df["human_response"].tolist(), options)
        m_dist  = empirical_dist(q_df[response_col].tolist(),    options)
        tvds.append(total_variation_distance(h_dist, m_dist, options))

    if not tvds:
        return float("nan"), 0
    return round(1.0 - sum(tvds) / len(tvds), 4), len(tvds)


def main() -> None:
    # ── 1. Check input files exist ────────────────────────────────────────────
    missing = [
        (BASELINE_PATH, "python run_baseline_opinion_qa.py"),
        (MODEL_PATH,    "python model_opinion_qa_v1.py"),
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
    if n_use < len(test_df):
        print(
            f"Warning: output files have fewer rows than test dataset "
            f"(baseline={len(baseline_responses)}, model={len(model_responses)}, "
            f"test={len(test_df)}). Using first {n_use} rows.\n"
        )

    test_df = test_df.iloc[:n_use].reset_index(drop=True)
    test_df["options"]            = test_df["options"].apply(json.loads)
    test_df["response_baseline"]  = baseline_responses.iloc[:n_use].values
    test_df["response_model"]     = model_responses.iloc[:n_use].values

    # n_parsed_b = test_df["response_baseline"].notna().sum()
    # n_parsed_m = test_df["response_model"].notna().sum()
    # print(f"Records used       : {n_use:,}")
    # print(f"Parsed (baseline)  : {n_parsed_b:,} ({n_parsed_b / n_use * 100:.1f}%)")
    # print(f"Parsed (model)     : {n_parsed_m:,} ({n_parsed_m / n_use * 100:.1f}%)\n")

    # ── 3. Per-persona representativeness ────────────────────────────────────
    summary_rows = []
    for persona_id, group in test_df.groupby("persona_id"):
        persona_desc = group["persona_description"].iloc[0]

        rep_b, n_q_b = compute_representativeness(group, "response_baseline")
        rep_m, n_q_m = compute_representativeness(group, "response_model")

        pct = round((rep_m - rep_b) * 100 / rep_b, 2) if rep_b and not np.isnan(rep_b) else float("nan")

        summary_rows.append({
            "persona_id":          persona_id,
            "persona_description": persona_desc,
            "n_questions":         n_q_b,
            "rep_baseline":        rep_b,
            "rep_model":           rep_m,
            "pct_improvement":     pct,
        })

        # print("=" * 72)
        # print(f"Persona ID  : {persona_id}")
        # print(f"Persona     : {persona_desc}")
        # print(f"Questions   : {n_q_b}/{n_q_m}")
        # print(f"Rep(g)      : baseline={rep_b}  model={rep_m}  pct_improvement={pct if not np.isnan(pct) else 'n/a'}")
        # print()

    if not summary_rows:
        print("No personas found in the input file.")
        return

    # ── 4. Build result table + Overall row ───────────────────────────────────
    result_df = pd.DataFrame(summary_rows).sort_values("persona_id").reset_index(drop=True)

    mean_b      = result_df["rep_baseline"].mean()
    mean_m      = result_df["rep_model"].mean()
    overall_pct = round((mean_m - mean_b) * 100 / mean_b, 2) if mean_b and not np.isnan(mean_b) else float("nan")

    result_df = pd.concat([result_df, pd.DataFrame([{
        "persona_id":          "Overall",
        "persona_description": "",
        "n_questions":         result_df["n_questions"].sum(),
        "rep_baseline":        round(mean_b, 4),
        "rep_model":           round(mean_m, 4),
        "pct_improvement":     overall_pct,
    }])], ignore_index=True)

    print("=" * 72)
    print("SUMMARY — ordered by persona_id")
    print("=" * 72)
    print(result_df.to_string(index=False))
    print()
    print(f"Mean Rep(g) (baseline) : {mean_b:.4f}")
    print(f"Mean Rep(g) (model)    : {mean_m:.4f}")
    print(f"Overall % improvement  : {overall_pct:.2f}%")

    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
