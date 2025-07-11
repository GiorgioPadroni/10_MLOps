import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


# Consts
MODEL_PATH = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
CONFIG = AutoConfig.from_pretrained(MODEL_PATH)

class FastText:
    def __init__(self, model_path=MODEL_PATH, tokenizer=TOKENIZER, config=CONFIG):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)


    def _preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    

    def _predict_sentiment(self, text):
        preprocessed_text = self._preprocess(text)
        encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        pred_label = self.config.id2label[ranking[0]]
        pred_score = scores[ranking[0]]

        return pred_label, np.round(float(pred_score), 4)

    
    def __call__(self, text):
        return self._predict_sentiment(text)