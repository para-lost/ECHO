#!/usr/bin/env python3
import os
import json
import pandas as pd
from collections import Counter
import eval  # your module
import numpy as np
# ---- Config ----
split = "text_to_image"
csv_file = "all_scores.csv"

judge_files = [
    f"./runs/gpt/{split}/{csv_file}",
    f"./runs/gemini/{split}/{csv_file}",
    f"./runs/qwen/{split}/{csv_file}",
]

def maybe_extract_score(df: pd.DataFrame) -> pd.Series:
    """
    If your CSV already has a numeric 'score' column, use it.
    If not, try evaluate_pairwise.extract_score(row) on a 'raw' column.
    """
    if "score" in df.columns:
        return pd.to_numeric(df["score"], errors="coerce")
    if "raw" in df.columns and hasattr(eval, "extract_score"):
        return df["raw"].apply(eval.extract_score)
    raise ValueError("No usable score source: expected 'score' column or 'raw'+extract_score")

def load_judge_results(judge_file: str) -> pd.DataFrame:
    """Load results from one judge CSV -> DataFrame[question_id, model, score]."""
    if not os.path.exists(judge_file):
        print(f"[WARN] Missing judge file: {judge_file}")
        return pd.DataFrame(columns=["question_id", "model", "score"])

    df = pd.read_csv(judge_file)
    # Normalize column names (just in case)
    df.columns = [c.strip() for c in df.columns]

    required = {"question_id", "model"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{judge_file} must contain columns: {required}")

    # Derive/clean score
    df["score"] = maybe_extract_score(df)
    df = df.dropna(subset=["score"])
    df = df[["question_id", "model", "score"]].copy()
    return df

def majority_vote_pairwise(judge_results_list):
    """
    1) Convert each judge's per-question scores to pairwise with scores_to_pairwise
       -> columns expected: question_id, model_a, model_b, winner
    2) Majority vote on winner among judges for each (qid, model_a, model_b)
    """
    pairwise_results = []
    for i, judge_results in enumerate(judge_results_list):
        if judge_results.empty:
            continue
        pairwise = eval.scores_to_pairwise(judge_results)
        pairwise["judge_id"] = i
        pairwise_results.append(pairwise)

    if not pairwise_results:
        return pd.DataFrame()

    all_pairwise = pd.concat(pairwise_results, ignore_index=True)

    def get_majority_vote_pairwise(group: pd.DataFrame) -> pd.Series:
        # votes for model_a/model_b
        a = group["model_a"].iloc[0]
        b = group["model_b"].iloc[0]
        votes_a = (group["winner"] == a).sum()
        votes_b = (group["winner"] == b).sum()

        if votes_a > votes_b:
            winner = a
        elif votes_b > votes_a:
            winner = b
        else:
            # Tie: use first judge as tiebreaker to keep deterministic
            winner = group["winner"].iloc[0]

        agreement = max(votes_a, votes_b) / len(group)
        return pd.Series({
            "question_id": group["question_id"].iloc[0],
            "model_a": a,
            "model_b": b,
            "winner": winner,
            "vote_count": len(group),
            "agreement": agreement
        })

    majority = (
        all_pairwise
        .groupby(["question_id", "model_a", "model_b"], as_index=False)
        .apply(get_majority_vote_pairwise)
        .reset_index(drop=True)
    )
    return majority

def main():
    # Load all judge CSVs
    print("Loading results from all judges...")
    judge_results = []
    for jf in judge_files:
        df = load_judge_results(jf)
        judge_results.append(df)
        print(f"Loaded {len(df)} rows from {jf}")

    # Determine common models from the Gemini judge (index 1)
    if len(judge_results) < 2 or judge_results[1].empty:
        raise RuntimeError("Gemini judge (index 1) missing or empty; cannot determine common models.")
    gemini_models = set(judge_results[1]["model"].unique())
    print(f"\nCommon models from gemini judge: {sorted(gemini_models)}")

    # Filter all judges to those models
    filtered = []
    for i, df in enumerate(judge_results):
        fdf = df[df["model"].isin(gemini_models)].copy()
        filtered.append(fdf)
        print(f"Judge {i}: kept {len(fdf)} rows after filtering to common models")

    # Majority vote across judges (pairwise)
    print("\nTaking majority vote on pairwise comparisons...")
    autoeval_results = majority_vote_pairwise(filtered)
    print(f"Majority vote results: {len(autoeval_results)} pairwise comparisons")
    if not autoeval_results.empty and "agreement" in autoeval_results.columns:
        print(f"Average agreement: {autoeval_results['agreement'].mean():.3f}")

    # Winrates (fastchat arena-style)
    print("\nComputing winrates...")
    winrate = eval.arena_fastchat_winrate(autoeval_results)
    # 'winrate' is typically a pandas DataFrame; print nicely
    try:
        print(winrate.sort_values("winrate", ascending=False))
    except Exception:
        print(winrate)
        
        # --- Std/SE consistent with arena_fastchat_winrate (per-comparison) ---

    # Ensure winner is in {'model_a','model_b'} or None (ties)
    df = autoeval_results.copy()
    # If your majority vote produced model names as winner, map back to side labels:
    if not df['winner'].isin(['model_a', 'model_b']).all():
        def to_side(row):
            w = row['winner']
            if pd.isna(w):
                return None
            if w == row['model_a']:
                return 'model_a'
            if w == row['model_b']:
                return 'model_b'
            # keep original if it's already a side, else None (treated as tie)
            return w if w in ('model_a', 'model_b') else None
        df['winner'] = df.apply(to_side, axis=1)

    a_win = df['winner'].eq('model_a')
    b_win = df['winner'].eq('model_b')
    score_a = np.where(a_win, 1.0, np.where(b_win, 0.0, 0.5))
    score_b = 1.0 - score_a

    scores = pd.concat([
        pd.DataFrame({'question_id': df['question_id'], 'model': df['model_a'], 'score': score_a}),
        pd.DataFrame({'question_id': df['question_id'], 'model': df['model_b'], 'score': score_b}),
    ], ignore_index=True)

    per_model = scores.groupby('model')['score']
    print(per_model)
    winrate_cmp = per_model.mean()                # matches arena winrate
    std_cmp     = per_model.std(ddof=1)           # sample std over comparisons
    N_cmp       = per_model.count()               # number of comparisons for the model
    se_cmp      = std_cmp / np.sqrt(N_cmp)        # empirical SE on same unit
    ci_low      = winrate_cmp - 1.96 * se_cmp
    ci_high     = winrate_cmp + 1.96 * se_cmp

    arena_consistent = (
        pd.DataFrame({
            'model': winrate_cmp.index,
            'winrate': winrate_cmp.values,
            'std': std_cmp.values,
            'N': N_cmp.values,
            'stderr': se_cmp.values,
            'ci95_low': ci_low.values,
            'ci95_high': ci_high.values,
        })
        .sort_values('winrate', ascending=False)
        .reset_index(drop=True)
    )

    # Merge with your previously computed `winrate` table if you want to keep extra columns
    try:
        winrate_with_se = winrate.merge(arena_consistent, on='model', how='left', suffixes=('', '_arena'))
    except Exception:
        winrate_with_se = arena_consistent

    print("\nArena-consistent std/SE over per-comparison scores:")
    print(winrate_with_se)

    out_path = "model_winrates_with_stderr_arena_interleaved.csv"
    winrate_with_se.to_csv(out_path, index=False)
    print(f"Saved arena-consistent stats to {out_path}")
    return per_model


if __name__ == "__main__":
    per_model = main()
    
