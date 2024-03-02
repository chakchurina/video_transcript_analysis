import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

from app.analytics.base_processor import BaseTextProcessor
from app.analytics.sentiment_analyzer import SentimentAnalyzer
from app.analytics.summarizer import TextSummarizer


class InsightExtractor(BaseTextProcessor):
    """
    Extracts insights such as outstanding sentences or summaries from a given DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the InsightExtractor with a DataFrame and applies sentiment analysis.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df

        self.analyzer = SentimentAnalyzer()
        self.summarizer = TextSummarizer()

    def emotional_messages(self, n=10) -> List[int]:
        """
        Identifies the top non-neutral sentences based on emotion scores.

        Returns:
            List[int]: Indices of top non-neutral sentences.
        """
        top_non_neutral_indices = sorted(self.df['emotion_score'].nlargest(n).index.tolist())
        return top_non_neutral_indices

    def questions(self, n=10) -> List[int]:
        """
        Filters sentences that are questions and returns indices of the most emotional
        questions if there are more than n questions.

        Returns:
            List[int]: Indices of the most emotional questions or all questions if 10 or less.
        """
        questions_df = self.df[self.df['question']]

        if len(questions_df) >= n:
            questions_df = questions_df.sort_values(by='emotion_score', ascending=False)
            question_indices = questions_df.index[:n].tolist()
        else:
            question_indices = questions_df.index.tolist()

        return question_indices

    def intros(self, n=10) -> List[int]:
        """
        Identifies sentences that resemble introductions.

        Returns:
            List[int]: Indices of sentences that look like intros.
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
        return indices[:n]

    def get_highlights(self, n) -> List[int]:
        """
        Combines emotional messages, questions, and intros to identify key insights.

        Returns:
            List[int]: Combined list of unique indices representing key insights.
        """
        self.analyzer.apply_to_dataframe(self.df)
        logging.info("Sentiment analysis applied to DataFrame.")

        emotionals = self.emotional_messages(n)
        questions = self.questions(n)
        intros = self.intros(n)

        highlights = list(set(emotionals + questions + intros))
        logging.info(f"Extracted {len(highlights)} highlights from the DataFrame: "
                     f"emotionals: {len(emotionals)}, questions: {len(questions)}, intros: {len(intros)}.")

        return highlights

    def get_summary(self, n) -> Tuple[List[str], List[int]]:
        """
        Get a summary of the text data contained within the DataFrame.

        Returns:
            Tuple[List[str], List[int]]: A tuple containing two elements:
                - A list of keywords extracted from the text data.
                - A list of indices of the sentences that form the summary.
        """
        keywords: List[str] = self.summarizer.get_keywords(self.df)
        summary: List[int] = self.summarizer.summarize(self.df, n)

        return keywords, summary
