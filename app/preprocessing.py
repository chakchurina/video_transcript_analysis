import os
import string
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from config.config import OPENAI_API_KEY, MODEL

LONG = 35  # Максимальная длина предложения для разделения
SHORT = 5  # Минимальная длина предложения для объединения


class DataProcessor:
    # todo: preprocessor looks ad-hoc, I'd change it to be more universal

    def __init__(self, data_path, file_names, file_index):
        self.file_path = os.path.join(data_path, file_names[file_index])
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.raw_df = pd.read_csv(self.file_path)
        self.raw_df.rename(columns={'length': 'time'}, inplace=True)

    @staticmethod
    def clean_tokenize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return words

    def add_aux_columns(self, df):
        df['embedding'] = df['sentence'].apply(self.get_embeddings)
        df['tokens'] = df['sentence'].apply(self.clean_tokenize)
        df['tempo'] = df['tokens'].apply(len) / df['time']
        df['length'] = df['tokens'].apply(len)
        df['question'] = df['sentence'].str.contains('\?')

        return df

    def add_time_codes(self, df):
        start_times, end_times = [0], []
        for i in range(1, len(df)):
            start_times.append(start_times[i - 1] + df.loc[i - 1, 'time'])

        df['start_time'] = start_times
        df['end_time'] = df['start_time'] + df['time']

        return df

    def get_embeddings(self, text):
        response = self.client.embeddings.create(input=text, model=MODEL)
        return response.data[0].embedding

    def get_cosine_distance(self, embeddings):
        cos_distances = [None]
        for i in range(1, len(embeddings)):
            cos_distance = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            cos_distances.append(cos_distance)
        return cos_distances

    def merge_sentences(self):
        self.raw_df = self.add_aux_columns(self.raw_df)
        self.raw_df['cos_dist'] = self.get_cosine_distance(self.raw_df['embedding'].tolist())

        similarity_cutoff = self.raw_df['cos_dist'].quantile(0.8)
        close_indices = self.raw_df.index[self.raw_df['cos_dist'] > similarity_cutoff].tolist()
        sentences, times = [self.raw_df.loc[0, 'sentence']], [self.raw_df.loc[0, 'time']]

        i = 1
        while i < len(self.raw_df):
            current, current_t, length = self.raw_df.loc[i, ['sentence', 'time', 'length']]
            previous, previous_t = sentences[-1], times[-1]
            if i in close_indices and length <= SHORT:
                sentences[-1], times[-1] = previous + " " + current, previous_t + current_t
            elif previous.endswith('...') and current.startswith('...'):
                sentences[-1], times[-1] = previous[:-3] + " " + current[3:], previous_t + current_t
            else:
                sentences.append(current), times.append(current_t)
            i += 1
        return pd.DataFrame({'sentence': sentences, 'time': times})

    def export_to_csv(self, df, filename):
        df.to_csv(filename, index=False)

    def prepare_data(self):
        df = self.merge_sentences()
        df = self.add_aux_columns(df)
        df = self.add_time_codes(df)
        return df
