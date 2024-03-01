import logging
import numpy as np
import pandas as pd

from typing import List
from app.analytics.sentiment_analyzer import SentimentAnalyzer
from app.analytics.base_processor import BaseTextProcessor


class InsightExtractor(BaseTextProcessor):
    """
    Extracts insights such as emotional messages, questions, and intros from a given DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the InsightExtractor with a DataFrame and applies sentiment analysis.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df
        self.analyzer = SentimentAnalyzer()
        self.analyzer.apply_to_dataframe(self.df)
        logging.info("Sentiment analysis applied to DataFrame.")

    def emotional_messages(self) -> List[int]:
        """
        Identifies the top non-neutral sentences based on emotion scores.

        Returns:
            List[int]: Indices of top non-neutral sentences.
        """
        top_non_neutral_indices = sorted(self.df['emotion_score'].nlargest(8).index.tolist())
        return top_non_neutral_indices

    def questions(self) -> List[int]:
        """
        Filters sentences that are questions.

        Returns:
            List[int]: Indices of sentences that are questions.
        """
        questions_df = self.df[self.df['question']]
        question_indices = questions_df.index.tolist()
        return question_indices

    def intros(self) -> List[int]:
        """
        Identifies sentences that resemble introductions, using a predefined request for embeddings comparison.

        Returns:
            List[int]: Indices of sentences that resemble introductions.
        """
        threshold = 0.765
        request = ("My name is Ankit Singla and I'm a full-time blogger. I blog about blogging. "
                   "I'm Karen, an entrepreneur and VC consultant. "
                   "Paul Erdős was a Hungarian mathematician. He was one of the most prolific "
                   "mathematicians and producers of mathematical conjectures of the 20th century. "
                   "This is Maria and she is ML Engineer at Rask")

        request_embedding = self.calculate_embeddings(request)
        request_embedding = np.array(request_embedding).reshape(1, -1)

        indices = self.threshold_closest(request_embedding, self.df, threshold=threshold)
        return indices

    def get_highlights(self) -> List[int]:
        """
        Combines emotional messages, questions, and intros to identify key insights.

        Returns:
            List[int]: Combined list of unique indices representing key insights.
        """
        emotionals = self.emotional_messages()
        questions = self.questions()
        intros = self.intros()

        highlights = list(set(emotionals + questions + intros))
        logging.info(f"Extracted {len(highlights)} highlights from the DataFrame.")
        return highlights
