from app.preprocessing import DataProcessor
from app.sentiment_analyzer import SentimentAnalyzer

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