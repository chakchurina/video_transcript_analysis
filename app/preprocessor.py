import os
import string
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from config.config import OPENAI_API_KEY, EMBEDDINGS_MODEL, EMBEDDINGS_PATH


class DataProcessor:
    def __init__(self, data_path, file):
        # todo измени то, что теперь мы работаем со ссылкой
        # todo измени, чтобы не надо было читать файл, если он в кэше
        self.file_path = os.path.join(data_path, file)

        self.df = pd.read_csv(self.file_path)
        self.df.rename(columns={'length': 'time'}, inplace=True)

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return words

    @staticmethod
    def calculate_embeddings(text):
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(input=text, model=EMBEDDINGS_MODEL)
        return response.data[0].embedding

    @staticmethod
    def get_cosine_distance(embeddings):
        cos_distances = [None]
        for i in range(1, len(embeddings)):
            cos_distance = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            cos_distances.append(cos_distance)
        return cos_distances

    def add_aux_columns(self, df):
        df['tokens'] = df['sentence'].apply(self.tokenize)
        df['tempo'] = df['tokens'].apply(len) / df['time']
        df['length'] = df['tokens'].apply(len)
        df['question'] = df['sentence'].str.contains('\?')

        # todo: found out late that there are pauses between phrases
        offset = 0.23
        df['time'] = df['time'] + offset

        df['start_time'] = df['time'].cumsum().shift(fill_value=0)
        df['end_time'] = df['start_time'] + df['time']

    def add_embeddings(self, df, video_id):
        embeddings_file = os.path.join(EMBEDDINGS_PATH, f'{video_id}.pkl')
        if os.path.exists(embeddings_file):
            print(f'Embeddings for {video_id} are already cached. Loading from pickle.')
            # todo сделай нормально
            with open(embeddings_file, 'rb') as file:
                df['embedding'] = pickle.load(file)
        else:
            print(f'Calculating embeddings for {video_id}.')
            df['embedding'] = df['sentence'].apply(self.calculate_embeddings)
            with open(embeddings_file, 'wb') as file:
                pickle.dump(df['embedding'].tolist(), file)

    def prepare_data(self, video_id):
        self.add_embeddings(self.df, video_id)
        self.add_aux_columns(self.df)
        return self.df
