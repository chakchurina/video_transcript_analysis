import os
import string
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from app.services.openai_service import calculate_embeddings

from config.config import EMBEDDINGS_PATH


class DataProcessor:
    def __init__(self, data_path, file, video_id):
        self.file_path = os.path.join(data_path, file)
        self.video_id = video_id
        self.df = None

    def process_data(self):
        self.df = pd.read_csv(self.file_path)

        self.df.rename(columns={'length': 'time'}, inplace=True)
        self.add_embeddings(self.video_id)

        # Add auxiliary columns for later analysis
        self.df['tokens'] = self.df['sentence'].apply(self.tokenize)
        self.df['tempo'] = self.df['tokens'].apply(len) / self.df['time']
        self.df['length'] = self.df['tokens'].apply(len)
        self.df['question'] = self.df['sentence'].str.contains('\?')

        # Calculate timecodes
        # todo: found out too late that there are pauses between phrases:(
        offset = 0.23
        self.df['time'] = self.df['time'] + offset
        self.df['start_time'] = self.df['time'].cumsum().shift(fill_value=0)
        self.df['end_time'] = self.df['start_time'] + self.df['time']

        return self.df

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return words

    @staticmethod
    def get_cosine_distance(embeddings):
        cos_distances = [None]
        for i in range(1, len(embeddings)):
            cos_distance = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            cos_distances.append(cos_distance)
        return cos_distances

    def add_embeddings(self, video_id):
        # todo add decent logging
        embeddings_file = os.path.join(EMBEDDINGS_PATH, f'{video_id}.pkl')

        if os.path.exists(embeddings_file):
            print(f'Embeddings for {video_id} are already cached. Loading from pickle.')

            with open(embeddings_file, 'rb') as file:
                self.df['embedding'] = pickle.load(file)
        else:
            print(f'Calculating embeddings for {video_id}.')
            self.df['embedding'] = self.df['sentence'].apply(calculate_embeddings)
            with open(embeddings_file, 'wb') as file:
                pickle.dump(self.df['embedding'].tolist(), file)
