import numpy as np
import math

from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import cosine_similarity


class TextSegmenter:
    def __init__(self, df):
        self.df = df

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

    def segment_text(self, p_size=10):  # todo replace p_size

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

    def get_context(self, n, size):
        paragraph = self.df.loc[n, 'segment']

        start_paragraph = max(0, paragraph - size)
        end_paragraph = paragraph + size  # todo учесть длину df

        return list(range(start_paragraph, end_paragraph))

    def get_cosine_closest_paragraphs(self, target_sentence_index, top_n=5):
        # Вычисляем косинусную схожесть между целевым предложением и всеми остальными
        target_embedding = np.array(self.df.loc[target_sentence_index, 'embedding']).reshape(1, -1)
        all_embeddings = np.array(self.df['embedding'].tolist())
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]

        # Находим индексы топ-N ближайших предложений
        closest_indices = np.argsort(similarities)[::-1][
                          1:top_n + 1]  # +1 чтобы исключить само целевое предложение

        closest_paragraphs = self.df.loc[closest_indices, 'segment'].unique().tolist()
        return closest_paragraphs
