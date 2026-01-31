# fed_agent/tools/fomc_tools.py

import re
import pandas as pd
from prompts import fomc_prompt

DATA_PATH = "data/communications.csv"


def _load_fomc_df() -> pd.DataFrame:
    """
    Load FOMC communications from CSV and normalize column names/types.
    Expected columns in CSV:
      - 'Date' or 'date'
      - 'Text' or 'text'
    """
    df = pd.read_csv(DATA_PATH)

    # Normalize date column name
    if "Date" in df.columns:
        date_col = "Date"
    elif "date" in df.columns:
        date_col = "date"
    else:
        raise ValueError("communications.csv must have a 'Date' or 'date' column.")

    # Normalize text column name
    if "Text" in df.columns:
        text_col = "Text"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError("communications.csv must have a 'Text' or 'text' column.")

    df = df.rename(columns={date_col: "date", text_col: "text"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "text"]).sort_values("date").reset_index(drop=True)
    return df


def _extract_years_and_position(user_query: str):
    """
    Parse the user query to extract:
    - list of years mentioned (sorted, unique)
    - position keyword: 'first', 'last', 'middle', or None
    """
    msg = user_query.lower()

    # Years like 1999, 2018, 2025, etc.
    years = sorted(
        {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", msg)}
    )

    # Position keywords
    first_words = ["first", "earliest", "start", "starting", "beginning"]
    last_words = ["last", "latest", "end", "final", "most recent"]
    middle_words = ["middle", "mid", "midpoint", "middle point", "mid-year", "mid year"]

    position = None
    if any(w in msg for w in first_words):
        position = "first"
    elif any(w in msg for w in last_words):
        position = "last"
    elif any(w in msg for w in middle_words):
        position = "middle"

    return years, position


def _select_meeting_row(df: pd.DataFrame, user_query: str) -> tuple[pd.Series | None, str | None]:
    """
    Core logic to choose which FOMC meeting to use based on the user query.

    Returns:
        (row, error_message)
        - row: a pandas Series for the selected meeting, or None
        - error_message: a string to return to the user if selection is ambiguous or impossible
    """
    years, position = _extract_years_and_position(user_query)
    msg = user_query.lower()

    min_date = df["date"].min()
    max_date = df["date"].max()
    min_year = min_date.year
    max_year = max_date.year

    # 1. If a specific year is mentioned
    if years:
        year = years[0]  # take the first year mentioned
        if year < min_year or year > max_year:
            return None, (
                f"I don't have any FOMC communications for {year}. "
                f"My dataset covers {min_year}–{max_year}."
            )

        df_year = df[df["date"].dt.year == year]
        if df_year.empty:
            return None, (
                f"I don't have any FOMC communications for {year}. "
                f"My dataset covers {min_year}–{max_year}."
            )

        # If the user did not specify first/last/middle, treat as ambiguous
        if position is None:
            n = len(df_year)
            return None, (
                f"I found {n} FOMC communications in {year}. "
                "Please specify whether you want the FIRST, LAST, or MIDDLE meeting in that year, "
                "or provide an exact date."
            )

        # Select within that year
        df_year = df_year.sort_values("date").reset_index(drop=True)
        if position == "first":
            row = df_year.iloc[0]
        elif position == "last":
            row = df_year.iloc[-1]
        elif position == "middle":
            row = df_year.iloc[len(df_year) // 2]
        else:
            # Shouldn't get here, but fallback to latest in year
            row = df_year.iloc[-1]

        return row, None

    # 2. No year mentioned → interpret global "first/last/middle" or default latest
    if position == "first":
        # "first ever meeting", "earliest meeting", "starting of the dataset"
        row = df.iloc[0]
        return row, None

    if position == "last":
        # "last meeting", "most recent meeting", etc.
        row = df.iloc[-1]
        return row, None

    if position == "middle":
        # "middle meeting", "midpoint meeting", etc. (across entire dataset)
        row = df.iloc[len(df) // 2]
        return row, None

    # 3. Special phrasing like "first ever meeting in history" but no year
    special_first = ["first ever", "first meeting ever"]
    if any(phrase in msg for phrase in special_first):
        row = df.iloc[0]
        return row, None

    # 4. Special phrasing like "last fed meeting", "latest fed meeting"
    special_last = ["last fed meeting", "latest fed meeting", "most recent meeting"]
    if any(phrase in msg for phrase in special_last):
        row = df.iloc[-1]
        return row, None

    # 5. Completely generic question about "the meeting" with no date hint:
    #    default to the latest meeting in the dataset (what you were doing before).
    row = df.iloc[-1]
    return row, None


def analyze_fomc_statement(llm, user_query: str) -> str:
    """
    Main entrypoint used by the agent.

    - Loads communications.csv
    - Detects the year & position (first/last/middle) requested
    - Selects the appropriate meeting or returns an informative message
    - Calls the LLM with fomc_prompt(statement_text, user_query)
    """
    df = _load_fomc_df()
    row, error_msg = _select_meeting_row(df, user_query)

    if error_msg is not None:
        # Ambiguous or out-of-range → return generic, non-hallucinated message
        return error_msg

    meeting_date = row["date"].date().isoformat()
    statement_text = row["text"]

    # Include the date explicitly in the context we send to the LLM
    context = f"Meeting date: {meeting_date}\n\n{statement_text}"

    prompt = fomc_prompt(context, user_query)
    response = llm.invoke(prompt)
    return response
