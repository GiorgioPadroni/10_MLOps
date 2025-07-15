# std libraries
import glob
import re
import time

# third party libraries
from datasets import Dataset
import pandas as pd
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch



def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
    

if __name__ == "__main__":
    # Caricamento modello e tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Cerca tutti i CSV nella cartella data
    csv_files = glob.glob("data/train/*.csv")

    if not csv_files:
        raise ValueError("No CSV files found in /data")
    
    # Carica e concatena tutti i CSV
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    ds = Dataset.from_pandas(df)
    
    # Mapping etichette PRIMA della tokenizzazione
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    ds = ds.map(lambda row: {"label": label_map[row["sentiment"]]})

    # Tokenizzazione DOPO (così 'text' è ancora presente)
    ds = ds.map(tokenize, batched=True)

    model_id = f"PonzioPilates97/sentiment_{int(time.time())}"

    # Setup Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        save_strategy="no",  # Nessun salvataggio per velocità
        push_to_hub=True,
        hub_model_id=model_id
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        # eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.push_to_hub(model_id)
    tokenizer.push_to_hub(model_id)