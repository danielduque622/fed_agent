"""
eval_kpis.py

Offline evaluation utilities for FedSentiment.

This file DOES NOT change the chatbot behavior. You run it separately
(e.g. `python eval_kpis.py`) to compute KPIs for your report/presentation.

Subsystem 1: RAG / Retrieval & Date Resolution
    - Recall@k
    - Precision@k
    - Mean Reciprocal Rank (MRR)
    - FOMC date-resolution accuracy (does natural-language date mapping pick
      the correct meeting?)
    - Embedding drift helper (compare retrieval across embedding systems)

Subsystem 2: LLM FOMC Summary / Tone
    - Tone prediction loop over REAL communications.csv (calls LLM in a loop)
    - Hallucination evaluation loop over REAL communications.csv (calls LLM)
    - Tone classification metrics (accuracy, F1, confusion) once you add labels
    - Hallucination rate (lexical heuristic)
    - Human evaluation rubric averages (0–5) once you fill scores

Subsystem 3: S&P 500 ML Forecast (ML)
    - MAPE (Mean Absolute Percentage Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - RMSE% and MAE% relative to average true close
    - Directional accuracy (up vs down)

IMPORTANT:
    - LLM-based KPIs require GOOGLE_API_KEY in your environment.
    - This script assumes it lives in the same directory as:
        agent.py, app.py, fomc_tools.py, sp500_tools.py, prompts.py, state.py
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import List, Dict, Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
)
from sklearn.linear_model import LinearRegression

from langchain_google_genai import GoogleGenerativeAI

# Project helpers
from fomc_tools import _load_fomc_df, _select_meeting_row
from sp500_tools import _prepare_sp500_df, add_return_lag_features
from prompts import fomc_prompt, sentiment_prompt

# Handle ResourceExhausted cleanly
try:
    from google.api_core.exceptions import ResourceExhausted
except Exception:  # fallback if import not available
    class ResourceExhausted(Exception):
        pass


# =============================================================================
# 0. LLM utilities (Gemini client + rate-limit-safe invoke)
# =============================================================================

def make_llm(model: str = "gemini-2.5-flash") -> Optional[GoogleGenerativeAI]:
    """
    Create the same Gemini client that Streamlit uses.

    Requires GOOGLE_API_KEY to be set in the environment.
    """
    api_key = ".."
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not set. Skipping LLM-based KPIs.")
        return None
    try:
        llm = GoogleGenerativeAI(model=model, google_api_key=api_key)
        return llm
    except Exception as e:
        print(f"⚠️  Error creating Gemini LLM client: {e}")
        return None


def safe_llm_invoke(llm, prompt: str, max_retries: int = 5, wait_seconds: int = 60):
    """
    Calls llm.invoke(prompt) with retry logic for Gemini 429 rate limits.

    - Catches ResourceExhausted (429 quota / rate-limit).
    - Waits `wait_seconds` (default 60) before retrying.
    - Retries up to `max_retries` times.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(prompt)
        except ResourceExhausted as e:
            msg = str(e).lower()
            if "exceeded your current quota" in msg or "429" in msg:
                print(f"\n⚠️  Gemini 429 (quota/rate limit) on attempt {attempt}/{max_retries}.")
                print(f"   Waiting {wait_seconds} seconds before retry...\n")
                time.sleep(wait_seconds)
                continue
            else:
                print(f"❌ ResourceExhausted (not quota-related): {e}")
                raise
        except Exception as e:
            print(f"❌ Non-429 error from LLM: {e}")
            raise

    raise RuntimeError(
        f"LLM call failed after {max_retries} retries due to repeated 429/quota errors."
    )


# =============================================================================
# 1. RETRIEVAL / RAG KPIs (Recall@k, Precision@k, MRR)
# =============================================================================

def recall_at_k(ground_truth_id, retrieved_ids: Sequence, k: int) -> float:
    """
    Recall@k for a single query.

    With exactly one relevant item (the correct meeting), Recall@k is:
        1.0 if correct meeting is in top-k retrieved, else 0.0
    """
    top_k = list(retrieved_ids)[:k]
    return 1.0 if ground_truth_id in top_k else 0.0


def precision_at_k(ground_truth_id, retrieved_ids: Sequence, k: int) -> float:
    """
    Precision@k for a single query.

    With exactly one relevant item, Precision@k is:
        1/k if the correct meeting is in top-k, else 0.0
    """
    top_k = list(retrieved_ids)[:k]
    if ground_truth_id in top_k:
        return 1.0 / float(k)
    return 0.0


def reciprocal_rank(ground_truth_id, retrieved_ids: Sequence) -> float:
    """
    Reciprocal Rank for a single query.

    If correct meeting is ranked at position r (1-based):
        RR = 1 / r
    If not found:
        RR = 0.
    """
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id == ground_truth_id:
            return 1.0 / idx
    return 0.0


def compute_retrieval_metrics(
    cases: List[Dict],
    k: int = 3,
) -> Dict[str, float]:
    """
    Compute Recall@k, Precision@k, and MRR over a list of retrieval cases.

    Each `case` dict MUST have:
        - 'ground_truth_id': the correct meeting identifier (e.g. "2018-08-01")
        - 'retrieved_ids'  : ordered list of meeting identifiers

    Example:
        cases = [
            {
                "ground_truth_id": "2018-08-01",
                "retrieved_ids": ["2018-08-01", "2018-06-13", "2018-09-26"],
            },
            ...
        ]
    """
    recalls = []
    precisions = []
    rrs = []

    for case in cases:
        gt = case["ground_truth_id"]
        retrieved = case["retrieved_ids"]

        recalls.append(recall_at_k(gt, retrieved, k))
        precisions.append(precision_at_k(gt, retrieved, k))
        rrs.append(reciprocal_rank(gt, retrieved))

    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
        "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
        "mrr": float(np.mean(rrs)) if rrs else float("nan"),
    }


# =============================================================================
# 1b. FOMC DATE-RESOLUTION KPIs (does date logic select the right meeting?)
# =============================================================================

def evaluate_fomc_date_resolution(verbose: bool = True) -> Dict[str, object]:
    """
    Test whether the _select_meeting_row() logic picks the correct FOMC meeting
    for natural-language queries like:

        - "Summarize latest FOMC meeting"
        - "Summarize the first FOMC meeting of 2018"
        - "Summarize starting of 2025 meeting"

    Uses the REAL communications.csv via _load_fomc_df().
    """
    df = _load_fomc_df()
    df = df.sort_values("date").reset_index(drop=True)

    min_year = int(df["date"].dt.year.min())
    max_year = int(df["date"].dt.year.max())
    latest_date = df["date"].iloc[-1].date()

    # Helpers
    def first_date_of_year(y: int):
        subset = df[df["date"].dt.year == y]
        return subset["date"].min().date() if not subset.empty else None

    def last_date_of_year(y: int):
        subset = df[df["date"].dt.year == y]
        return subset["date"].max().date() if not subset.empty else None

    cases = []

    # Latest overall
    cases.append({"query": "Summarize latest FOMC meeting", "expected_date": latest_date})
    cases.append({"query": "Summarize the last Fed meeting", "expected_date": latest_date})

    # Earliest year
    first_year = min_year
    fy_first = first_date_of_year(first_year)
    fy_last = last_date_of_year(first_year)
    if fy_first is not None:
        cases.append({
            "query": f"Summarize the first FOMC meeting of {first_year}",
            "expected_date": fy_first,
        })
        cases.append({
            "query": f"Summarize starting of {first_year} meeting",
            "expected_date": fy_first,
        })
    if fy_last is not None:
        cases.append({
            "query": f"Summarize the last FOMC meeting of {first_year}",
            "expected_date": fy_last,
        })

    # Latest year
    last_year = max_year
    ly_first = first_date_of_year(last_year)
    ly_last = last_date_of_year(last_year)
    if ly_first is not None:
        cases.append({
            "query": f"Summarize the first FOMC meeting of {last_year}",
            "expected_date": ly_first,
        })
        cases.append({
            "query": f"Summarize starting of {last_year} meeting",
            "expected_date": ly_first,
        })
    if ly_last is not None:
        cases.append({
            "query": f"Summarize the last FOMC meeting of {last_year}",
            "expected_date": ly_last,
        })

    results = []
    correct_count = 0

    for case in cases:
        query = case["query"]
        expected = case["expected_date"]

        row, err = _select_meeting_row(df, query)
        if err is not None or row is None:
            selected = None
        else:
            selected = row["date"].date()

        is_correct = (selected == expected)
        if is_correct:
            correct_count += 1

        results.append({
            "query": query,
            "expected_date": expected,
            "selected_date": selected,
            "correct": is_correct,
            "error": err,
        })

    accuracy = correct_count / len(results) if results else float("nan")

    if verbose:
        print("\n=== FOMC date-resolution evaluation ===")
        for r in results:
            print(f"Query: {r['query']}")
            print(f"  Expected: {r['expected_date']}")
            print(f"  Selected: {r['selected_date']}")
            print(f"  Correct:  {r['correct']}")
            if r["error"]:
                print(f"  Error:    {r['error']}")
            print()
        print(f"Overall accuracy: {accuracy:.3f}")

    return {
        "accuracy": accuracy,
        "cases": results,
    }


# =============================================================================
# 1c. Embedding drift helpers (for future embedding experiments)
# =============================================================================

def compute_embedding_drift_two_systems(
    cases_a: List[Dict],
    cases_b: List[Dict],
    k: int = 3,
) -> Dict[str, float]:
    """
    Compare retrieval consistency between two embedding systems (Gemini vs HF vs OpenAI).

    Each list element:
        - 'query': the user query
        - 'retrieved_ids': ordered list of meeting IDs for that query

    cases_a[i] and cases_b[i] must correspond to the SAME query.
    """
    if len(cases_a) != len(cases_b):
        raise ValueError("cases_a and cases_b must have the same length (same queries).")

    jaccards = []
    disagree_top1 = []

    for case_a, case_b in zip(cases_a, cases_b):
        ids_a = list(case_a["retrieved_ids"])[:k]
        ids_b = list(case_b["retrieved_ids"])[:k]

        set_a = set(ids_a)
        set_b = set(ids_b)
        if not set_a and not set_b:
            continue

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union > 0 else 0.0
        jaccards.append(jaccard)

        # Disagreement on top-1
        top1_a = ids_a[0] if ids_a else None
        top1_b = ids_b[0] if ids_b else None
        disagree_top1.append(0.0 if top1_a == top1_b else 1.0)

    return {
        "avg_jaccard_at_k": float(np.mean(jaccards)) if jaccards else float("nan"),
        "top1_disagreement_rate": float(np.mean(disagree_top1)) if disagree_top1 else float("nan"),
    }


# =============================================================================
# 2. LLM FOMC SUMMARY / TONE KPIs
# =============================================================================

def compute_tone_classification_metrics(
    labels_csv_path: str,
    true_col: str = "true_tone",
    pred_col: str = "pred_tone",
) -> Dict[str, object]:
    """
    Compute classification metrics for Hawkish / Neutral / Dovish tone.

    You need a CSV with at least two columns:
        - true_tone: labels you (human) assigned
        - pred_tone: labels produced by your LLM
    """
    df = pd.read_csv(labels_csv_path)

    y_true = df[true_col].astype(str).str.strip()
    y_pred = df[pred_col].astype(str).str.strip()

    labels = sorted(y_true.unique().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def _tokenize_words(text: str) -> List[str]:
    """
    Very simple tokenizer: lowercases and extracts alphabetic word tokens.
    """
    text = text.lower()
    return re.findall(r"[a-z]+", text)


def compute_hallucination_rate_from_pairs(
    contexts: List[str],
    answers: List[str],
    threshold: float = 0.20,
) -> Dict[str, float]:
    """
    Core hallucination metric using lexical overlap over lists of (context, answer).
    """
    fractions = []
    halluc_flags = []

    for context, answer in zip(contexts, answers):
        ctx_words = set(_tokenize_words(context))
        ans_words = _tokenize_words(answer)

        if not ans_words:
            continue

        ans_word_set = set(ans_words)
        new_words = ans_word_set - ctx_words

        frac = len(new_words) / max(1, len(ans_word_set))
        fractions.append(frac)
        halluc_flags.append(frac > threshold)

    if not fractions:
        return {
            "hallucination_rate": float("nan"),
            "avg_new_word_fraction": float("nan"),
            "threshold": threshold,
        }

    halluc_rate = float(np.mean(halluc_flags))
    avg_new_frac = float(np.mean(fractions))

    return {
        "hallucination_rate": halluc_rate,
        "avg_new_word_fraction": avg_new_frac,
        "threshold": threshold,
    }


def run_fomc_tone_predictions_over_dataset(
    llm,
    max_rows: Optional[int] = 30,
    out_path: str = "data/eval_fomc_tone_predictions.csv",
) -> pd.DataFrame:
    """
    LOOP THE REAL DATASET + CALL LLM to get tone predictions.

    Steps:
        1) Load communications.csv via _load_fomc_df().
        2) Take the last `max_rows` meetings (or all if max_rows is None).
        3) For each meeting:
            - Take an excerpt of the statement text.
            - Build sentiment_prompt(excerpt).
            - Call safe_llm_invoke(llm, prompt).
            - Parse the 'Tone: ...' line.
        4) Save results to CSV with columns:
            ['date', 'text_excerpt', 'raw_output', 'pred_tone'].

    Then you can manually add a 'true_tone' column and run
    compute_tone_classification_metrics(...) on that CSV.
    """
    df = _load_fomc_df().sort_values("date").reset_index(drop=True)

    if max_rows is not None and max_rows > 0:
        df_eval = df.tail(max_rows).copy()
    else:
        df_eval = df.copy()

    records = []

    for _, row in df_eval.iterrows():
        date_str = row["date"].date().isoformat()
        text = str(row["text"])
        excerpt = text[:1000]  # first 1000 characters

        prompt = sentiment_prompt(excerpt)
        resp = safe_llm_invoke(llm, prompt)
        resp_text = getattr(resp, "content", resp)

        pred_tone = "Unknown"
        for line in str(resp_text).splitlines():
            if line.lower().startswith("tone:"):
                pred_tone = line.split(":", 1)[1].strip()
                break

        records.append({
            "date": date_str,
            "text_excerpt": excerpt,
            "raw_output": resp_text,
            "pred_tone": pred_tone,
        })

        # Proactive throttle to avoid 429 (Gemini free tier: 10 RPM)
        time.sleep(7)

    out_df = pd.DataFrame(records)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved tone predictions for {len(out_df)} meetings to {out_path}")
    print("Tone distribution:")
    print(out_df["pred_tone"].value_counts())

    return out_df


def run_fomc_hallucination_eval_over_dataset(
    llm,
    max_rows: int = 20,
    out_path: str = "data/eval_fomc_hallucination_auto.csv",
    threshold: float = 0.20,
) -> Dict[str, float]:
    """
    LOOP THE REAL DATASET + CALL LLM to estimate hallucination rate.

    Steps:
        1) Load communications.csv via _load_fomc_df().
        2) Take the last `max_rows` meetings.
        3) For each:
            - Build context = full statement text (with date).
            - Ask: "Summarize this FOMC meeting."
            - Build fomc_prompt(context, question).
            - Call safe_llm_invoke(llm, prompt).
        4) Compute lexical hallucination heuristic.
        5) Save context + answer pairs to CSV.

    Returns:
        dict with hallucination_rate, avg_new_word_fraction, threshold.
    """
    df = _load_fomc_df().sort_values("date").reset_index(drop=True)
    df_eval = df.tail(max_rows).copy()

    contexts = []
    answers = []
    rows_out = []

    for _, row in df_eval.iterrows():
        date_str = row["date"].date().isoformat()
        text = str(row["text"])

        context = f"Meeting date: {date_str}\n\n{text}"
        user_q = "Summarize this FOMC meeting."

        prompt = fomc_prompt(context, user_q)
        resp = safe_llm_invoke(llm, prompt)
        answer = getattr(resp, "content", resp)

        contexts.append(text)  # raw FOMC text only for lexical overlap
        answers.append(str(answer))

        rows_out.append({
            "date": date_str,
            "context": text,
            "answer": str(answer),
        })

        # Proactive throttle to avoid 429
        time.sleep(7)

    out_df = pd.DataFrame(rows_out)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved hallucination eval pairs for {len(out_df)} meetings to {out_path}")

    metrics = compute_hallucination_rate_from_pairs(contexts, answers, threshold=threshold)
    print("Hallucination metrics (lexical heuristic):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def compute_human_eval_scores(
    csv_path: str,
    content_col: str = "content_score",
    halluc_col: str = "hallucination_score",
    tone_col: str = "tone_match_score",
) -> Dict[str, float]:
    """
    Compute averages for a 0–5 human evaluation rubric.

    You create a CSV with columns:

        statement_id,content_score,hallucination_score,tone_match_score
        2018-08-01,4,5,5
        ...

    Returns average scores for each dimension.
    """
    df = pd.read_csv(csv_path)

    content = df[content_col].astype(float)
    halluc = df[halluc_col].astype(float)
    tone = df[tone_col].astype(float)

    return {
        "avg_content_score": float(content.mean()),
        "avg_hallucination_score": float(halluc.mean()),
        "avg_tone_match_score": float(tone.mean()),
    }


# =============================================================================
# 3. S&P 500 ML FORECAST KPIs (MAPE, RMSE, MAE, directional accuracy)
# =============================================================================

def evaluate_sp500_ml(
    test_fraction: float = 0.2,
    max_lag: int = 5,
    clip_daily_move: float = 0.05,
) -> Dict[str, float]:
    """
    Evaluate the S&P 500 ML model using a simple train/test split on log-returns.

    Uses YOUR existing helpers: _prepare_sp500_df() and add_return_lag_features().

    Returns:
        - num_test_points
        - mape_percent
        - rmse
        - mae
        - rmse_pct  (RMSE as % of avg true close in test)
        - mae_pct   (MAE as % of avg true close in test)
        - directional_accuracy
    """
    df = _prepare_sp500_df()
    if len(df) < 50:
        raise ValueError("Not enough S&P 500 data to evaluate ML model (need >= 50 rows).")

    df_feat = add_return_lag_features(df, max_lag=max_lag)

    feature_cols = [c for c in df_feat.columns if c.startswith("lag_")]
    X = df_feat[feature_cols].values
    y_logret = df_feat["log_return"].values
    closes = df_feat["close"].values

    n = len(df_feat)
    test_size = max(1, int(math.floor(test_fraction * n)))
    train_size = n - test_size

    X_train = X[:train_size]
    y_train = y_logret[:train_size]
    X_test = X[train_size:]
    y_test = y_logret[train_size:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict log returns on test set
    y_pred_logret = model.predict(X_test)
    y_pred_logret = np.clip(y_pred_logret, -clip_daily_move, clip_daily_move)

    # Reconstruct predicted closes vs actual closes
    true_closes = []
    pred_closes = []
    true_dirs = []
    pred_dirs = []

    for i in range(test_size):
        idx = train_size + i
        prev_close = closes[idx - 1]
        true_logret = y_test[i]
        pred_logret = y_pred_logret[i]

        true_close = prev_close * math.exp(true_logret)
        pred_close = prev_close * math.exp(pred_logret)

        true_closes.append(true_close)
        pred_closes.append(pred_close)

        true_dirs.append(1 if true_logret >= 0 else 0)
        pred_dirs.append(1 if pred_logret >= 0 else 0)

    true_closes = np.array(true_closes)
    pred_closes = np.array(pred_closes)

    # MAPE (%)
    ape = np.abs((true_closes - pred_closes) / true_closes) * 100.0
    mape = float(np.mean(ape))

    # RMSE & MAE (in index points)
    rmse = float(math.sqrt(mean_squared_error(true_closes, pred_closes)))
    mae = float(mean_absolute_error(true_closes, pred_closes))

    # RMSE% & MAE% relative to average true close
    avg_true_close = float(true_closes.mean())
    if avg_true_close > 0:
        rmse_pct = rmse / avg_true_close * 100.0
        mae_pct = mae / avg_true_close * 100.0
    else:
        rmse_pct = float("nan")
        mae_pct = float("nan")

    # Directional accuracy
    true_dirs = np.array(true_dirs)
    pred_dirs = np.array(pred_dirs)
    dir_acc = float(np.mean(true_dirs == pred_dirs))

    return {
        "num_test_points": int(test_size),
        "mape_percent": mape,
        "rmse": rmse,
        "mae": mae,
        "rmse_pct": rmse_pct,
        "mae_pct": mae_pct,
        "directional_accuracy": dir_acc,
        "avg_true_close_test": avg_true_close,
    }


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    print("\n=== FedSentiment KPI Evaluation ===\n")

    # 3) S&P 500 ML forecast KPIs (NO LLM NEEDED)
    print("=== S&P 500 ML forecast evaluation ===")
    try:
        ml_metrics = evaluate_sp500_ml()
        for k, v in ml_metrics.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Error evaluating ML model:", e)

    # 1b) FOMC date-resolution KPIs (NO LLM NEEDED)
    try:
        _ = evaluate_fomc_date_resolution(verbose=True)
    except Exception as e:
        print("Error evaluating FOMC date resolution:", e)

    # LLM-based KPIs (tone + hallucination loops over REAL dataset)
    llm = make_llm()
    if llm is not None:
        # 2a) Tone predictions over real dataset
        try:
            run_fomc_tone_predictions_over_dataset(
                llm,
                max_rows=30,
                out_path="data/eval_fomc_tone_predictions.csv",
            )
        except Exception as e:
            print("Error running FOMC tone predictions:", e)

        # 2b) Hallucination evaluation over real dataset
        try:
            run_fomc_hallucination_eval_over_dataset(
                llm,
                max_rows=20,
                out_path="data/eval_fomc_hallucination_auto.csv",
                threshold=0.20,
            )
        except Exception as e:
            print("Error running FOMC hallucination evaluation:", e)

    print("\nDone.\n")
