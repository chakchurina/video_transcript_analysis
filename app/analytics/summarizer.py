import logging
from typing import List, Set
import pandas as pd
from collections import Counter

from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words

from app.analytics.base_processor import BaseTextProcessor


class TextSummarizer(BaseTextProcessor):
    """
    Provides functionalities for text summarization and keyword extraction.
    """

    def __init__(self, language: str = 'english') -> None:
        """
        Initializes the text summarizer with the specified language and its stop words.

        Args:
            language (str): The language of the text to be summarized. Defaults to 'english'.
        """
        self.language: str = language
        self.stop_words: Set[str] = set(get_stop_words(language.upper()))
        logging.info(f"TextSummarizer initialized for {language} language.")

    def summarize(self, df: pd.DataFrame, sentences_count: int = 10) -> List[int]:
        """
        Summarizes the text contained in a DataFrame using extractive summarization.

        Args:
            df (pd.DataFrame): DataFrame containing sentences to summarize.
            sentences_count (int): Number of sentences for the summary.

        Returns:
            List[int]: Indices of sentences included in the summary.
        """
        text: str = ' '.join(df['sentence'].tolist())

        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count=sentences_count)

        sentence_numbers: List[int] = []
        for sentence in summary:
            indices = [i for i, s in enumerate(df['sentence'].tolist()) if str(sentence) == s]
            sentence_numbers.extend(indices)

        logging.info("Text summarized successfully.")
        return sorted(list(set(sentence_numbers)))

    def get_keywords(self, df: pd.DataFrame, top_n: int = 3) -> List[str]:
        """
        Extracts keywords from the text contained in a DataFrame using a Bag-of-Words approach.

        Args:
            df (pd.DataFrame): DataFrame containing sentences from which to extract keywords.
            top_n (int): Number of top keywords to return.

        Returns:
            List[str]: A list of extracted keywords.
        """
        keywords: List[str] = []
        text: str = ' '.join(df['sentence'].tolist())

        words: List[str] = self.tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        word_counts: Counter = Counter(words)
        most_common_words: List[tuple] = word_counts.most_common(top_n)
        keywords.extend([word[0] for word in most_common_words])

        logging.info(f"Top {top_n} keywords extracted.")
        return keywords
