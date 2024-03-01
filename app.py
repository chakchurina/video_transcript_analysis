import pandas as pd

from app.data_processor import DataProcessor
from app.analytics.sentiment_analyzer import SentimentAnalyzer
from app.analytics.insight_extractor import InsightExtractor
from app.analytics.summarizer import TextSummarizer
from app.analytics.segmenter import TextSegmenter
from app.services.youtube_service import YouTubeService
from app.services.openai_service import prompt_gpt  # todo refactor

from config.config import TEXTS_PATH, VIDEOS


if __name__ == "__main__":

    # todo change the way of storing it
    TRANSCRIPT, LINK = VIDEOS[4]
    TRANSCRIPT = TRANSCRIPT + '.csv'  # todo cringe

    # get video and comments from YouTube
    youtube = YouTubeService()
    video_id = youtube.extract_video_id(LINK)
    youtube.download_video(video_id=video_id)
    comments = youtube.get_comments(video_id)

    # preprocess transcript and add auxiliary columns
    df = DataProcessor(TEXTS_PATH, TRANSCRIPT, video_id)

    # try to enrich data using comments (disabled for now)
    # comments_df = pd.DataFrame(comments)

    text = ' '.join(df['sentence'])

    # todo Sentiment Analysis move it to Insights Extractor

    analyzer = SentimentAnalyzer()
    analyzer.apply_to_dataframe(df, 'sentence')

    # todo Sentiment Analysis

    extractor = InsightExtractor(df)
    emotional = extractor.emotional_messages()
    fast_responses = extractor.fast_phrases()
    qa_pairs = extractor.question_answer_pairs()
    intros = extractor.extract_intros()

    segmenter = TextSegmenter(df)
    segmenter.segment_text(p_size=10)

    # todo: сделать так, что этот кусок будет возвращать предложения

    # Пример использования:
    summarizer = TextSummarizer()

    # Резюмирование текста
    summary = summarizer.summarize(text, 3)
    print(summary)

    # Получения ключевых слов темы текста
    theme_keywords = summarizer.get_keywords(df)

    # todo: get the most commented part of video

    context_consecutive = segmenter.get_context(emotional[0], size=1)
    context_closest = segmenter.get_cosine_closest_paragraphs(emotional[0], top_n=3)
    context = set(context_consecutive + context_closest)

    context_df = df[df['segment'].isin(context)]
    context_string = "\n".join([f"{index}: {row['sentence']}" for index, row in context_df.iterrows()])

    selected = prompt_gpt(emotional[0], context_string, theme_keywords)
    generated_text = ' '.join(df.loc[selected, 'sentence'])

    # cutter = VideoEditor(video_id)
    # cutter.cut_sentences_from_video(df, selected)

    print(generated_text)
