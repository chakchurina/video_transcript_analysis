import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessor import DataProcessor


class InsightExtractor:
    def __init__(self, df):
        self.df = df

    def extract_emotional_messages(self, emotional_threshold=0.5):
        self.df['emotional'] = self.df['emotion_score'] > emotional_threshold
        return self.df[self.df['emotional']]

    def extract_fast_responses(self, tempo_threshold=0.75):
        fastest = self.df['tempo'].quantile(tempo_threshold)
        return self.df[
            (self.df['emotion_score'] > self.df['emotion_score'].quantile(0.5)) & (self.df['tempo'] > fastest)]

    def find_question_answer_pairs(self):
        questions_df = self.df[self.df['question'] == True]
        statements_df = self.df[self.df['question'] == False]

        closest_statements = {}
        for index, question_embedding in questions_df.iterrows():
            similarities = cosine_similarity([question_embedding['embedding']], list(statements_df['embedding']))
            top_5_indices = similarities[0].argsort()[-3:][::-1]
            closest_sentences = statements_df.iloc[top_5_indices]['sentence'].values
            closest_statements[question_embedding['sentence']] = closest_sentences

        return closest_statements

    def find_intros(df):
        similarity_threshold = 0.765
        # todo ad-hoc threshold, very sorry

        request = "My name is Ankit Singla and I'm a full-time blogger. I blog about blogging. " \
                  "I'm Karen, an entrepreneur and VC consultant. " \
                  "Paul Erdős was a Hungarian mathematician. He was one of the most prolific " \
                  "mathematicians and producers of mathematical conjectures of the 20th century. " \
                  "This is Maria and she is a Data Engineer at Rask"
        request_embedding = get_embeddings(request)
        request_embedding = np.array(request_embedding).reshape(1, -1)  # Подготавливаем вектор запроса

        sentence_similarities = []
        for index, row in df.iterrows():
            embedding = np.array(row['embedding']).reshape(1, -1)  # Подготавливаем вектор предложения
            similarity = cosine_similarity(embedding, request_embedding)[0][0]
            if similarity > similarity_threshold:
                sentence_similarities.append((index, row['sentence'], similarity))

        sorted_sentences = sorted(sentence_similarities, key=lambda x: x[2], reverse=True)
        print(sorted_sentences)
        return [{i: sentence} for i, sentence, _ in sorted_sentences]

    intros = find_intros(df)
    print(intros)

