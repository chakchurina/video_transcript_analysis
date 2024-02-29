from app.preprocessor import DataProcessor
from app.sentiment_analyzer import SentimentAnalyzer
from app.insight_extractor import InsightExtractor
from app.summarizer import TextSummarizer

from config.config import TEXTS_PATH, FILE_NAMES, FILE


if __name__ == "__main__":

    processor = DataProcessor(TEXTS_PATH, FILE_NAMES, FILE)
    df = processor.prepare_data()

    text = ' '.join(df['sentence'])
    # mean_tempo = df['tempo'].mean()

    analyzer = SentimentAnalyzer()
    analyzer.apply_to_dataframe(df, 'sentence')
    print(df["emotion_score"])

    extractor = InsightExtractor(df)
    emotional_messages = extractor.extract_emotional_messages()
    fast_responses = extractor.extract_fast_responses()
    qa_pairs = extractor.find_question_answer_pairs()
    intros = extractor.find_intros()

    # Пример использования:
    summarizer = TextSummarizer()

    # Для резюмирования текста
    summary = summarizer.summarize_with_textrank(text, 3)
    print(summary)

    # Для получения ключевых слов темы текста
    theme_keywords = summarizer.get_text_theme_keywords(df['sentence'].tolist(), df['embedding'].tolist())

    # todo: get the most commented part of video

    print(intros)