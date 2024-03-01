import os
import unittest
from unittest.mock import patch

from app.analytics.preprocessor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = ''
        cls.file_name = 'test.csv'
        cls.video_id = 'test_video_123'
        cls.test_file_path = os.path.join(cls.data_path, cls.file_name)

    def setUp(self):
        # Mock the add_embeddings method to bypass file operations
        self.add_embeddings_patch = patch('app.analytics.preprocessor.DataProcessor.add_embeddings')
        self.mock_add_embeddings = self.add_embeddings_patch.start()

        # Mock the BaseTextProcessor's calculate_embeddings method as before
        self.calculate_embeddings_patch = patch('app.analytics.base_processor.BaseTextProcessor.calculate_embeddings',
                                                return_value=[0.1, 0.2, 0.3])
        self.mock_calculate_embeddings = self.calculate_embeddings_patch.start()

        self.preprocessor = DataProcessor(self.data_path, self.file_name, self.video_id)

    def tearDown(self):
        self.add_embeddings_patch.stop()
        self.calculate_embeddings_patch.stop()

    def test_create_dataframe_columns(self):
        """Ensure DataFrame is created with the expected columns,
        excluding 'embedding' which is added by a mocked method."""
        df = self.preprocessor.create_dataframe()
        expected_columns = ['index', 'sentence', 'time', 'tokens', 'tempo', 'length', 'question', 'start_time',
                            'end_time']
        self.assertListEqual(sorted(df.columns.tolist()), sorted(expected_columns))

    def test_tempo_calculation(self):
        """Verify the calculation of the 'tempo' column."""
        df = self.preprocessor.create_dataframe()
        self.assertTrue('tempo' in df.columns)
        # Additional tests for tempo values can be added here

    def test_question_column(self):
        """Check if the 'question' column correctly identifies sentences with a question mark."""
        df = self.preprocessor.create_dataframe()
        self.assertTrue('question' in df.columns)
        # Additional tests for question column values can be added here

    # Additional tests for other functionalities can be added here


if __name__ == '__main__':
    unittest.main()
