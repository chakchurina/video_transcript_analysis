import numpy as np
import math
import logging

from typing import List
from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame

from app.analytics.base_processor import BaseTextProcessor


class TextSegmenter(BaseTextProcessor):
    """
    Class for segmenting text into meaningful blocks based on sentence embeddings.
    """

    def __init__(self, df: DataFrame) -> None:
        """
        Initializes the TextSegmenter with a DataFrame and segments the text.

        Args:
            df (DataFrame): The DataFrame containing text data and embeddings.
        """
        self.df: DataFrame = df
        self.segment_text(p_size=10)

    def get_n_closest(self, sentence_index: int, n: int) -> List[int]:
        """
        Finds n closest sentences to a given sentence index based on cosine similarity.

        Args:
            sentence_index (int): The index of the sentence in the DataFrame.
            n (int): The number of closest sentences to find.

        Returns:
            List[int]: List of indices for the closest sentences.
        """
        sentence_embedding: np.ndarray = np.array(self.df.loc[sentence_index, 'embedding']).reshape(1, -1)
        closest_indexes: List[int] = self.top_n_closest(sentence_embedding, self.df, n)
        closest_paragraphs: List[int] = self.df.loc[closest_indexes, 'segment'].unique().tolist()
        context_indices: List[int] = self.df[self.df['segment'].isin(closest_paragraphs)].index.tolist()

        logging.info(f"Context indices for sentence {sentence_index}: {context_indices}")
        return context_indices

    def get_consecutive(self, sentence_number: int, back: int = 2, forward: int = 2) -> List[int]:
        """
        Finds consecutive sentences around a given sentence, assuming semantic relation.

        Args:
            sentence_number (int): The index of the central sentence.
            back (int): Number of sentences to look back.
            forward (int): Number of sentences to look forward.

        Returns:
            List[int]: List of indices for the consecutive sentences.
        """
        paragraph: int = self.df.loc[sentence_number, 'segment']
        start_paragraph: int = max(paragraph - back, 0)
        end_paragraph: int = paragraph + forward

        context: List[int] = self.df.loc[self.df['segment'].between(start_paragraph, end_paragraph)].index.tolist()
        return sorted(context)

    def rev_sigmoid(self, x: float) -> float:
        """
        Reverse sigmoid function used for weighting similarities.

        Args:
            x (float): The input value.

        Returns:
            float: The output of the reverse sigmoid function.
        """
        return 1 / (1 + math.exp(0.5 * x))

    def activate_similarities(self, similarities: np.ndarray, p_size: int = 10) -> np.ndarray:
        """
        Applies an activation function to the similarities to highlight significant segments.

        Args:
            similarities (np.ndarray): The cosine similarity matrix.
            p_size (int): The size of the paragraph to consider.

        Returns:
            np.ndarray: Activated similarities highlighting significant text blocks.
        """
        x: np.ndarray = np.linspace(-10, 10, p_size)
        y: np.ndarray = np.vectorize(self.rev_sigmoid)(x)

        activation_weights: np.ndarray = np.pad(y, (0, similarities.shape[0] - p_size), 'constant')
        diagonals: List[np.ndarray] = [similarities.diagonal(each) for each in range(1, similarities.shape[0])]
        diagonals: List[np.ndarray] = [np.pad(each, (0, similarities.shape[0] - len(each)), 'constant') for each in
                                       diagonals]
        diagonals: np.ndarray = np.stack(diagonals) * activation_weights[:len(diagonals)].reshape(-1, 1)
        activated_similarities: np.ndarray = np.sum(diagonals, axis=0)

        return activated_similarities

    def segment_text(self, p_size: int = 10) -> None:
        """
        Segments the text into paragraphs based on embeddings and similarity activations.

        Args:
            p_size (int): The size of the paragraph to consider for activation.
        """
        embeddings_matrix: np.ndarray = np.array(self.df['embedding'].tolist())
        cosine_sim_matrix: np.ndarray = cosine_similarity(embeddings_matrix)

        activated_similarities: np.ndarray = self.activate_similarities(cosine_sim_matrix, p_size=p_size)
        minimas: np.ndarray = argrelextrema(activated_similarities, np.less, order=2)

        split_points: List[int] = [each for each in minimas[0]]

        segment_numbers: List[int] = []
        segment_number: int = 0
        for num in range(len(self.df)):
            if num in split_points:
                segment_number += 1
            segment_numbers.append(segment_number)

        self.df['segment'] = segment_numbers
        logging.info("Text segmented successfully.")
