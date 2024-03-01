import numpy as np
import math

from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import cosine_similarity

from app.analytics.base_processor import BaseTextProcessor


class TextSegmenter(BaseTextProcessor):
    def __init__(self, df):
        # To find sentence context, first split the text, and then get the relevant blocks
        self.df = df
        self.segment_text(p_size=10)

    def get_n_closest(self, sentence_index, n):
        # Use cosine distance to find other relevant parts
        sentence_embedding = np.array(self.df.loc[sentence_index, 'embedding']).reshape(1, -1)

        closest_indexes = self.top_n_closest(sentence_embedding, self.df, n)
        closest_paragraphs = self.df.loc[closest_indexes, 'segment'].unique().tolist()
        context_indices = self.df[self.df['segment'].isin(closest_paragraphs)].index.tolist()

        # todo log it
        # context_string = "\n".join([f"{index}: {row['sentence']}" for index, row in context_df.iterrows()])
        return context_indices

    def get_consecutive(self, sentence_number, back=2, forward=2):
        # Assuming that consecutive paragraphs have semantic relation
        paragraph = self.df.loc[sentence_number, 'segment']

        # todo cover with units
        start_paragraph = paragraph - back
        end_paragraph = paragraph + forward

        context = self.df.loc[self.df['segment'].between(start_paragraph, end_paragraph)].index.tolist()
        return list(sorted(context))

    def rev_sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(0.5 * x))

    def activate_similarities(self, similarities: np.array, p_size=10) -> np.array:

        x = np.linspace(-10, 10, p_size)
        y = np.vectorize(self.rev_sigmoid)

        activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size), 'constant')
        diagonals = [similarities.diagonal(each) for each in range(1, similarities.shape[0])]
        diagonals = [np.pad(each, (0, similarities.shape[0] - len(each)), 'constant') for each in diagonals]
        diagonals = np.stack(diagonals)
        diagonals = diagonals * activation_weights[:diagonals.shape[0]].reshape(-1, 1)
        activated_similarities = np.sum(diagonals, axis=0)

        return activated_similarities

    def segment_text(self, p_size=10):

        embeddings_matrix = np.array(self.df['embedding'].tolist())
        cosine_sim_matrix = cosine_similarity(embeddings_matrix)

        activated_similarities = self.activate_similarities(cosine_sim_matrix, p_size=p_size)
        minimas = argrelextrema(activated_similarities, np.less, order=2)

        split_points = [each for each in minimas[0]]

        segment_number = 0
        segment_numbers = []

        for num in range(len(self.df)):
            if num in split_points:
                segment_number += 1
            segment_numbers.append(segment_number)

        self.df['segment'] = segment_numbers
