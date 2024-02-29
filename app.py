from app.preprocessor import DataProcessor
from app.sentiment_analyzer import SentimentAnalyzer
from app.insight_extractor import InsightExtractor

from config.config import DATA_PATH, FILE_NAMES, FILE


if __name__ == "__main__":
    processor = DataProcessor(DATA_PATH, FILE_NAMES, FILE)
    df = processor.prepare_data()

    print("\n".join(df['sentence']))
    print(df.head())

    mean_tempo = df['tempo'].mean()

    analyzer = SentimentAnalyzer()
    analyzer.apply_to_dataframe(df, 'sentence')
    print(df["emotion_score"])

    extractor = InsightExtractor(df)
    emotional_messages = extractor.extract_emotional_messages()
    fast_responses = extractor.extract_fast_responses()
    qa_pairs = extractor.find_question_answer_pairs()
    intros = extractor.find_intros()

    print(intros)