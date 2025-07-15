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
    metrics_path = "CI_CD/results/metrics.csv"
    metrics_df.to_csv(metrics_path)

    mlflow.set_tracking_uri("http://localhost:5000")  # oppure Docker se lo usi così
    mlflow.set_experiment("Model Evaluation")

    with mlflow.start_run():
        # Parametri
        mlflow.log_param("model_id", latest_model_id)
        mlflow.log_param("num_test_samples", len(df))

        # Logga alcune metriche globali
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])

        # Artefatto CSV
        mlflow.log_artifact(metrics_path)