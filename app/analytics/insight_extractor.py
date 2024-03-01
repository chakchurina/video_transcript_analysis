import numpy as np

from app.analytics.sentiment_analyzer import SentimentAnalyzer
from app.analytics.base_processor import BaseTextProcessor


class InsightExtractor(BaseTextProcessor):
    def __init__(self, df):
        self.df = df

        # highlight non-neutral sentences
        self.analyzer = SentimentAnalyzer()
        self.analyzer.apply_to_dataframe(self.df)

    def emotional_messages(self):
        top_non_neutral_indices = sorted(self.df['emotion_score'].nlargest(8).index.tolist())
        return top_non_neutral_indices

    def questions(self):
        questions_df = self.df[self.df['question']]
        question_indices = questions_df.index.tolist()
        return question_indices

    def intros(self):
        # one more ad-hoc threshold, very sorry
        threshold = 0.765

        # hack to get all intro-like sentences
        request = "My name is Ankit Singla and I'm a full-time blogger. I blog about blogging. " \
                  "I'm Karen, an entrepreneur and VC consultant. " \
                  "Paul Erdős was a Hungarian mathematician. He was one of the most prolific " \
                  "mathematicians and producers of mathematical conjectures of the 20th century. " \
                  "This is Maria and she is a Data Engineer at Rask"

        request_embedding = self.calculate_embeddings(request)
        request_embedding = np.array(request_embedding).reshape(1, -1)

        indices = self.threshold_closest(request_embedding, self.df, threshold=threshold)
        return indices

    def get_highlights(self):
        emotionals = self.emotional_messages()
        questions = self.questions()
        intros = self.intros()

        return list(set(emotionals + questions + intros))
