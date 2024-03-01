import os
import pandas as pd
import pickle
import logging

from app.analytics.base_processor import BaseTextProcessor
from config.config import EMBEDDINGS_PATH


class DataProcessor(BaseTextProcessor):
    """
    Prepares data for analytics.
    """

    def __init__(self, data_path: str, file: str, video_id: str) -> None:
        """
        Initializes the DataProcessor with paths and identifiers.

        Args:
            data_path (str): The path to the data directory.
            file (str): The filename of the dataset.
            video_id (str): The unique identifier for the video.
        """
        self.file_path: str = os.path.join(data_path, f"{file}.csv")
        self.video_id: str = video_id
        self.df: pd.DataFrame = pd.DataFrame()

    def create_dataframe(self) -> pd.DataFrame:
        """
        Creates and preprocesses the dataframe from the CSV file.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        self.df = pd.read_csv(self.file_path)

        self.df.rename(columns={'length': 'time'}, inplace=True)
        self.add_embeddings(self.video_id)

        self.df['tokens'] = self.df['sentence'].apply(self.tokenize)
        self.df['tempo'] = self.df['tokens'].apply(len) / self.df['time']
        self.df['length'] = self.df['tokens'].apply(len)
        self.df['question'] = self.df['sentence'].str.contains('\?')

        # todo: found out too late that there are pauses between phrases:(
        offset = 0.22
        self.df['time'] += offset
        self.df['start_time'] = self.df['time'].cumsum().shift(fill_value=0)
        self.df['end_time'] = self.df['start_time'] + self.df['time']

        return self.df

    def add_embeddings(self, video_id: str) -> None:
        """
        Adds embeddings to the dataframe, either by loading from a file or calculating them.

        Args:
            video_id (str): The video identifier for which embeddings are added.
        """
        embeddings_file: str = os.path.join(EMBEDDINGS_PATH, f'{video_id}.pkl')

        if os.path.exists(embeddings_file):
            logging.info(f'Embeddings for {video_id} are already cached. Loading from pickle.')
            with open(embeddings_file, 'rb') as file:
                self.df['embedding'] = pickle.load(file)
        else:
            logging.info(f'Calculating embeddings for {video_id}.')
            self.df['embedding'] = self.df['sentence'].apply(lambda x: self.calculate_embeddings(x))
            with open(embeddings_file, 'wb') as file:
                pickle.dump(self.df['embedding'].tolist(), file)
