import pandas as pd
from collections import defaultdict

def load_eval_data(path):
    df = pd.read_excel(path)
    grouped = defaultdict(list)

    for _, row in df.iterrows():
        grouped[row["Query"]].strip()
        grouped[row["Query"]].append(row["Assessment_url"])

    return [
        {"query": q, "relevant_urls": urls}
        for q, urls in grouped.items()
    ]
