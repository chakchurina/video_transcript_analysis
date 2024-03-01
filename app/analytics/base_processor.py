from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import string
import numpy as np

from config.config import OPENAI_API_KEY, EMBEDDINGS_MODEL


class BaseTextProcessor:
    client = OpenAI(api_key=OPENAI_API_KEY)

    def calculate_embeddings(self, text):
        response = self.client.embeddings.create(input=text, model=EMBEDDINGS_MODEL)
        return response.data[0].embedding

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return words

    @staticmethod
    def top_n_closest(target_embedding, df, n=3):
        # todo: check if actually works
        # Вычисляем расстояние между целевым предложением и всеми остальными
        all_embeddings = np.array(df['embedding'].tolist())
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]

        # Находим индексы топ-N ближайших предложений
        # todo упрости
        closest_indices = np.argsort(similarities)[::-1][1:n + 1]  # +1 исключает само целевое предложение
        return closest_indices

    @staticmethod
    def threshold_closest(target_embedding, df, threshold=0.7):
        sentence_similarities = []
        for index, row in df.iterrows():
            # Getting text embeddings
            embedding = np.array(row['embedding']).reshape(1, -1)
            similarity = cosine_similarity(embedding, target_embedding)[0][0]
            if similarity > threshold:
                sentence_similarities.append((index, similarity))

        sorted_sentences = sorted(sentence_similarities, key=lambda x: x[1], reverse=True)
        return [i for i, _ in sorted_sentences]
