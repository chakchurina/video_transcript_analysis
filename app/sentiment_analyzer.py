from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

from config.config import SENTIMENT_MODEL


class SentimentAnalyzer:
    # todo See if there's something I can throw away

    def __init__(self):
        self.model_name = SENTIMENT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        scores = softmax(logits, dim=1)
        # todo refactor 
        scores_dict = {label: score.item() for label, score in zip(['negative', 'neutral', 'positive'], scores[0])}
        return scores_dict

    def apply_to_dataframe(self, df, text_column):
        non_neutrals, positives, negatives = [], [], []

        for text in df[text_column]:
            sentiment_scores = self.predict_sentiment(text)
            non_neutrals.append(1 - sentiment_scores['neutral'])
            positives.append(sentiment_scores['positive'])
            negatives.append(sentiment_scores['negative'])

        df['emotion_score'] = non_neutrals
        df['positive_score'] = positives
        df['negative_score'] = negatives
