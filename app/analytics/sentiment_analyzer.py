import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

from config.config import SENTIMENT_MODEL


class SentimentAnalyzer:
    """
    Sentiment analysis class using transformers to predict sentiment scores for texts.
    """

    def __init__(self) -> None:
        """
        Initializes the SentimentAnalyzer with the specified sentiment model.
        """
        self.model_name: str = SENTIMENT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        logging.info("SentimentAnalyzer model and tokenizer loaded.")

    def predict_sentiment(self, text: str) -> dict:
        """
        Predicts sentiment scores for a given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary with sentiment scores for negative, neutral, and positive sentiments.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        scores = softmax(logits, dim=1)
        scores_dict = {label: score.item() for label, score in zip(['negative', 'neutral', 'positive'], scores[0])}
        return scores_dict

    def apply_to_dataframe(self, df: pd.DataFrame) -> None:
        """
        Applies sentiment analysis to all sentences in a DataFrame and adds sentiment scores.

        Args:
            df (pd.DataFrame): The DataFrame containing the sentences to analyze.
        """
        non_neutrals, positives, negatives = [], [], []

        for text in df['sentence']:
            sentiment_scores = self.predict_sentiment(text)
            non_neutrals.append(1 - sentiment_scores['neutral'])
            positives.append(sentiment_scores['positive'])
            negatives.append(sentiment_scores['negative'])

        df['emotion_score'] = non_neutrals
        df['positive_score'] = positives
        df['negative_score'] = negatives
        logging.info("Sentiment scores applied to DataFrame.")
