import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessor import DataProcessor


class InsightExtractor:
    def __init__(self, df):
        self.df = df

    def emotional_messages(self):
        top_non_neutral_indices = sorted(self.df['emotion_score'].nlargest(8).index.tolist())
        return top_non_neutral_indices

    def fast_phrases(self, tempo_threshold=0.75):
        fastest = self.df['tempo'].quantile(tempo_threshold)
        return self.df[
            (self.df['emotion_score'] > self.df['emotion_score'].quantile(0.5)) & (self.df['tempo'] > fastest)]

    def question_answer_pairs(self):
        # todo refactor
        questions_df = self.df[self.df['question'] == True]
        statements_df = self.df[self.df['question'] == False]

        closest_statements = {}
        for index, question_embedding in questions_df.iterrows():
            similarities = cosine_similarity([question_embedding['embedding']], list(statements_df['embedding']))
            top_5_indices = similarities[0].argsort()[-3:][::-1]
            closest_sentences = statements_df.iloc[top_5_indices]['sentence'].values
            closest_statements[question_embedding['sentence']] = closest_sentences

        return closest_statements

    def extract_intros(self):
        similarity_threshold = 0.765
        # one more ad-hoc threshold, very sorry

        # hack to get all intro-like sentences
        request = "My name is Ankit Singla and I'm a full-time blogger. I blog about blogging. " \
                  "I'm Karen, an entrepreneur and VC consultant. " \
                  "Paul Erdős was a Hungarian mathematician. He was one of the most prolific " \
                  "mathematicians and producers of mathematical conjectures of the 20th century. " \
                  "This is Maria and she is a Data Engineer at Rask"

        request_embedding = DataProcessor.calculate_embeddings(request)
        # Getting request embedding
        request_embedding = np.array(request_embedding).reshape(1, -1)

        sentence_similarities = []
        for index, row in self.df.iterrows():
            # Getting text embeddings
            embedding = np.array(row['embedding']).reshape(1, -1)
            similarity = cosine_similarity(embedding, request_embedding)[0][0]
            if similarity > similarity_threshold:
                sentence_similarities.append((index, row['sentence'], similarity))

        sorted_sentences = sorted(sentence_similarities, key=lambda x: x[2], reverse=True)
        print(sorted_sentences)
        return [{i: sentence} for i, sentence, _ in sorted_sentences]
