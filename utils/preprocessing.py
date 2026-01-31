# fed_agent/utils/preprocessing.py
import pandas as pd

def clean_text(text: str) -> str:
    return " ".join(str(text).split())

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df
