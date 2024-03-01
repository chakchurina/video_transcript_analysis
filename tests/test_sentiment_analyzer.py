import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from app.analytics.sentiment_analyzer import SentimentAnalyzer
from config.config import SENTIMENT_MODEL


class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        # Mocking the sentiment model initialization to avoid actual model loading
        self.init_patch = patch('app.analytics.sentiment_analyzer.AutoModelForSequenceClassification.from_pretrained')
        self.mock_init = self.init_patch.start()

        # Mocking the tokenizer initialization
        self.tokenizer_patch = patch('app.analytics.sentiment_analyzer.AutoTokenizer.from_pretrained')
        self.mock_tokenizer = self.tokenizer_patch.start()

        # Mocking the logging.info call
        self.logging_patch = patch('app.analytics.sentiment_analyzer.logging.info')
        self.mock_logging = self.logging_patch.start()

        self.sentiment_analyzer = SentimentAnalyzer()

    def tearDown(self):
        self.init_patch.stop()
        self.tokenizer_patch.stop()
        self.logging_patch.stop()

    def test_init(self):
        """Test the initialization of the SentimentAnalyzer."""
        self.mock_init.assert_called_once_with(SENTIMENT_MODEL)
        self.mock_tokenizer.assert_called_once_with(SENTIMENT_MODEL)
        self.mock_logging.assert_called_once()

    def test_apply_to_dataframe(self):
        """Test the apply_to_dataframe method."""
        # Mock a sample DataFrame
        df = pd.DataFrame({'sentence': ['This is a test sentence.', 'Another test sentence.']})

        # Mocking the predict_sentiment method to avoid actual sentiment prediction
        self.sentiment_analyzer.predict_sentiment = MagicMock(return_value={'negative': 0.1, 'neutral': 0.2, 'positive': 0.7})

        self.sentiment_analyzer.apply_to_dataframe(df)

        # Check if sentiment scores are added to the DataFrame
        self.assertTrue('emotion_score' in df.columns)
        self.assertTrue('positive_score' in df.columns)
        self.assertTrue('negative_score' in df.columns)

        # Check if the sentiment prediction method was called for each sentence
        self.assertEqual(self.sentiment_analyzer.predict_sentiment.call_count, 2)


if __name__ == '__main__':
    unittest.main()
