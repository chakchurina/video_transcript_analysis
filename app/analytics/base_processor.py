import string
import numpy as np

from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from config.config import OPENAI_API_KEY, EMBEDDINGS_MODEL


class BaseTextProcessor:
    """
    A base class for implementing text processing functionalities.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    def calculate_embeddings(self, text):
        """
        Calculates embeddings for a given text using a specified model.

        Args:
            text: The input text for which embeddings are calculated.

        Returns:
            The embeddings for the given text.
        """
        response = self.client.embeddings.create(input=text, model=EMBEDDINGS_MODEL)
        return response.data[0].embedding

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes a given text into words, removing punctuation and converting to lowercase.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of words from the text.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return words

    @staticmethod
    def top_n_closest(target_embedding, df, n=3):
        """
        Identifies the top N closest sentences to a target sentence based on embeddings.

        Args:
            target_embedding: The embedding of the target sentence.
            df: A DataFrame containing embeddings of other sentences to compare against.
            n: The number of closest sentences to return.

        Returns:
            Indices of the top N closest sentences in the DataFrame.
        """
        all_embeddings = np.array(df['embedding'].tolist())
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]

        closest_indices = np.argsort(similarities)[::-1][1:n + 1]  # + 1 excludes the target sentence
        return closest_indices

    @staticmethod
    def threshold_closest(target_embedding, df, threshold=0.7):
        """
        Finds sentences closer than a specified similarity threshold to a target sentence.

        Args:
            target_embedding: The embedding of the target sentence.
            df: A DataFrame containing embeddings for comparison.
            threshold: The similarity threshold for considering a sentence as close.

        Returns:
            A list of indices for sentences in df that are above the similarity threshold.
            But still no longer than 10
        """
        sentence_similarities = []
        for index, row in df.iterrows():
            embedding = np.array(row['embedding']).reshape(1, -1)
            similarity = cosine_similarity(embedding, target_embedding)[0][0]
            if similarity > threshold:
                sentence_similarities.append((index, similarity))

        sorted_sentences = sorted(sentence_similarities, key=lambda x: x[1], reverse=True)
        return [i for i, _ in sorted_sentences][:10]
