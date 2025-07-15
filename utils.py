# std libraries
import re
import requests

user = "PonzioPilates97"
PRETRAINED_MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

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

