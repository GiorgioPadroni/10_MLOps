# std libraries
import glob
import os

# third party libraries
from datasets import Dataset
import mlflow
import pandas as pd
from sklearn.metrics import classification_report
from transformers import pipeline, AutoTokenizer
import torch

# local libraries
from utils import get_latest_model_id, user, PRETRAINED_MODEL



latest_model_id = get_latest_model_id(user)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)


if __name__ == "__main__":
    pipe = pipeline("text-classification", model=latest_model_id, tokenizer=tokenizer)

    # Cerca tutti i CSV nella cartella data
    csv_files = glob.glob("data/test/*.csv")

    if not csv_files:
        raise ValueError("No CSV files found in /data")

    # Carica e concatena tutti i CSV
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    preds = pipe(df["text"].tolist(), truncation=True)
    predicted_labels = [p["label"].lower() for p in preds]

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y_true = df["sentiment"].map(label_map)
    y_pred = pd.Series(predicted_labels).map(label_map)

    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()

    os.makedirs("CI_CD/results", exist_ok=True)
    metrics_df.to_csv("CI_CD/results/metrics.csv")
    