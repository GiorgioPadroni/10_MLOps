import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Caricamento modello e tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

if __name__ == "__main__":
    # Caricamento dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    dataset = ds.map(tokenize, batched=True)

    # Setup Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        # evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        save_steps=500,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=f"PonzioPilates97/sentiment_{time.time()}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.push_to_hub()
    tokenizer.push_to_hub()
