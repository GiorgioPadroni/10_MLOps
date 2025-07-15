import glob
import re
import time

# third party libraries
from datasets import Dataset
import pandas as pd
import requests
from sklearn.metrics import classification_report
from transformers import pipeline
import torch


def get_latest_model_id(user: str, prefix: str = "sentiment_") -> str:
    url = f"https://huggingface.co/api/models?author={user}"
    response = requests.get(url)
    response.raise_for_status()
    models = response.json()

    # Filtra modelli che iniziano con il prefix
    filtered = [
        m["modelId"]
        for m in models
        if m["modelId"].startswith(f"{user}/{prefix}")
    ]

    if not filtered:
        raise ValueError("No models found with the given prefix.")

    # Ordina per timestamp (assunto nel nome)
    latest = max(filtered, key=lambda x: int(re.findall(rf"{prefix}(\d+)$", x)[0]))
    return latest


user = "PonzioPilates97"
latest_model_id = get_latest_model_id(user)


if __name__ == "__main__":
    pipe = pipeline("text-classification", model=latest_model_id, tokenizer=latest_model_id)

    # Cerca tutti i CSV nella cartella data
    csv_files = glob.glob("data/test/*.csv")

    if not csv_files:
        raise ValueError("No CSV files found in /data")

    # Carica e concatena tutti i CSV
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    preds = pipe(df["text"].tolist(), truncation=True)

    # Extract predicted labels
    predicted_labels = [p["label"].lower() for p in preds]

    # Convert to numerical format if needed
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y_true = df["sentiment"].map(label_map)
    y_pred = pd.Series(predicted_labels).map(label_map)

    # Compute classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Flatten to CSV
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv("CI_CD/results/metrics.csv")
