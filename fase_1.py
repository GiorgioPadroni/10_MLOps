# third party libraries
from datasets import load_dataset
from tqdm import tqdm

# local libraries
from models.FastText import FastText


def evaluate_dataset(ds, model):
    results = {v:0 for v in model.config.id2label.values()}
    for row in tqdm(ds):
        text = row["text"]
        pred, _ = model(text)
        results[pred] += 1
    return results

if __name__ == "__main__":
    ds = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test[:100]")
    model = FastText()
    results = evaluate_dataset(ds, model)
    print("Sentiment Distribution (su 100 tweet):", results)