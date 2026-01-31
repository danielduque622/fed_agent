# fed_agent/tools/sp500_tools.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from prompts import sp500_prompt

SP500_PATH = "data/SP500.csv"
FEDFUNDS_PATH = "data/FEDFUNDS.csv"
FOMC_PATH = "data/communications.csv"


def analyze_sp500(llm, user_query: str) -> str:
    """
    Qualitative S&P 500 outlook using the latest FOMC communication,
    latest FEDFUNDS rate, and recent S&P 500 trend.

    This is the LLM-based "market strategist" view.
    """
    sp = pd.read_csv(SP500_PATH)
    fed = pd.read_csv(FEDFUNDS_PATH)
    fomc = pd.read_csv(FOMC_PATH)

    fomc_summary = fomc["Text"].iloc[-1][:600]
    fed_rate = fed["FEDFUNDS"].iloc[-1]
    sp_trend = f"Recent trend shows last 5 closes: {sp['SP500'].tail(5).to_list()}"

    prompt = sp500_prompt(fomc_summary, fed_rate, sp_trend, user_query)
    return llm.invoke(prompt)


# ---------------------------------------------------------------------------
# ML FORECAST PIPELINE (IMPROVED)
# ---------------------------------------------------------------------------

def _prepare_sp500_df() -> pd.DataFrame:
    """
    Load S&P 500 CSV and standardize columns to:
        - 'date'
        - 'close'

    Handles the common schema: columns ['date', 'SP500'].

    If no 'date' column exists, creates a simple integer index as a placeholder.
    """
    df = pd.read_csv(SP500_PATH).copy()

    # Standardize close column
    if "SP500" in df.columns:
        df = df.rename(columns={"SP500": "close"})

    # Ensure numeric close
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Standardize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        # Fallback: create a dummy index as "date"
        df["date"] = pd.RangeIndex(len(df))

    # Drop rows with missing close
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df


def add_return_lag_features(df_sp500: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Create lag features on LOG RETURNS instead of raw levels.

    Steps:
    - Compute log_return_t = log(close_t / close_{t-1})
    - Create lag_1, ..., lag_max_lag on these log returns.
    """
    df = df_sp500.copy().sort_values("date")

    # Compute log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Build lag features on log_return
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["log_return"].shift(lag)

    # Drop rows with NaNs introduced by shifting
    df = df.dropna().reset_index(drop=True)
    return df


def train_sp500_linear_model(df_sp500: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame]:
    """
    Train a linear regression model to predict the NEXT log return
    using lagged log returns as features.

    Then we can forecast the next closing price as:
        close_{t+1} = close_t * exp(predicted_log_return)
    """
    # You can increase max_lag if you have enough data (e.g., 5, 10).
    df_feat = add_return_lag_features(df_sp500, max_lag=5)

    feature_cols = [c for c in df_feat.columns if c.startswith("lag_")]
    X = df_feat[feature_cols].values
    y = df_feat["log_return"].values  # predict next log return

    model = LinearRegression()
    model.fit(X, y)

    return model, df_feat


def predict_sp500_next_ml(model: LinearRegression, df_feat: pd.DataFrame) -> dict:
    """
    Use the last row in df_feat to forecast the NEXT log return,
    then map it back to a price level for the day AFTER the
    latest date in the S&P data.

    We ALSO clip the predicted log return to avoid insane moves
    (> ±5% daily), so the demo doesn't explode with >90% crashes.
    """
    df_sorted = df_feat.sort_values("date")
    last_row = df_sorted.iloc[-1]

    feature_cols = [c for c in df_feat.columns if c.startswith("lag_")]
    X_last = last_row[feature_cols].to_numpy().reshape(1, -1)

    # Predicted next log return
    pred_log_ret = float(model.predict(X_last)[0])

    # Optional but highly recommended: clip to -5%..+5%
    pred_log_ret = float(np.clip(pred_log_ret, -0.05, 0.05))

    last_close = float(last_row["close"])
    predicted_next_close = float(last_close * np.exp(pred_log_ret))

    return {
        "last_date": last_row["date"],
        "last_close": last_close,
        "predicted_next_close": predicted_next_close,
    }


def ml_forecast_sp500() -> str:
    """
    Public ML forecast function used by the agent.

    - Loads S&P 500 data
    - Trains the improved log-return regression model
    - Predicts the next day's close (AFTER latest date in SP500.csv)
    - Returns a human-readable string, compatible with the existing agent code.
    """
    df = _prepare_sp500_df()

    # Not enough data? Bail out gracefully.
    if len(df) < 20:
        return "Not enough S&P 500 data to train the ML model."

    model, df_feat = train_sp500_linear_model(df)
    forecast = predict_sp500_next_ml(model, df_feat)

    last_date = forecast["last_date"]
    last_close = forecast["last_close"]
    next_close = forecast["predicted_next_close"]

    # Format date nicely if it's a Timestamp
    if hasattr(last_date, "date"):
        last_date_str = last_date.date().isoformat()
    else:
        last_date_str = str(last_date)

    return (
        f"Last available S&P 500 date: {last_date_str}\n"
        f"Last close: {last_close:.2f}\n"
        f"Predicted next S&P 500 level (ML model): {next_close:.2f}"
    )
