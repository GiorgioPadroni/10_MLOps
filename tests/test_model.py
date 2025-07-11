from datasets import load_dataset
from models.FastText import FastText
from tqdm import tqdm
import unittest


ds = load_dataset("cardiffnlp/tweet_eval", "emotion", split="test[:100]")

class TestFastText(unittest.TestCase):
    def setUp(self):
        self.model = FastText()

    def test_prediction_output(self):
        for row in tqdm(ds):
            text = "I love this product!"
            label, score = self.model(text)
            self.assertIn(label.lower(), ["positive", "neutral", "negative"])
            self.assertTrue(0 <= score <= 1)

    def test_preprocess(self):
        text = "@user http://example.com great!"
        clean = self.model._preprocess(text)
        self.assertIn("@user", clean)
        self.assertIn("http", clean)


if __name__ == "__main__":
    unittest.main()