from collections import Counter
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
from sklearn.cluster import KMeans

from app.analytics.base_processor import BaseTextProcessor


class TextSummarizer(BaseTextProcessor):
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(get_stop_words(language.upper()))

    def summarize(self, df, sentences_count=10):
        # Extractive summarization
        text = ' '.join(df['sentence'].tolist())

        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count=sentences_count)

        sentence_numbers = []
        for sentence in summary:
            indices = [i for i, s in enumerate(df['sentence'].tolist()) if str(sentence) == s]
            sentence_numbers.extend(indices)

        return sorted(list(set(sentence_numbers)))

    def get_keywords(self, df, num_clusters=1):  # todo remove cluster names
        # todo тут не надо считать KMeans
        sentences = df['sentence'].tolist()
        embeddings = df['embedding'].tolist()

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(np.array(embeddings))
        cluster_labels = kmeans.labels_

        cluster_sentences = [[] for _ in range(num_clusters)]
        for i, sentence in enumerate(sentences):
            cluster_sentences[cluster_labels[i]].append(sentence)

        cluster_keywords = []
        for cluster in cluster_sentences:
            cluster_text = ' '.join(cluster)
            cluster_words = self.tokenize(cluster_text)
            cluster_words = [word for word in cluster_words if word not in self.stop_words]
            word_counts = Counter(cluster_words)
            most_common_words = word_counts.most_common(3)
            cluster_keywords.extend([word[0] for word in most_common_words])

        return cluster_keywords
