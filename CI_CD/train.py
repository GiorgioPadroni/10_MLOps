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

    # Subset piccolo e casuale per velocità
    train_dataset = dataset["train"].shuffle(seed=42).select(range(500))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(100))

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
        hub_model_id=f"PonzioPilates97/sentiment_{int(time.time())}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
