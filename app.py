import pandas as pd

from app.preprocessor import DataProcessor
from app.sentiment_analyzer import SentimentAnalyzer
from app.insight_extractor import InsightExtractor
from app.summarizer import TextSummarizer
from app.segmenter import TextSegmenter
from app.youtube_service import YouTubeService
from app.gpt_selector import prompt_gpt  # todo refactor
from app.video_editor import VideoEditor

from config.config import TEXTS_PATH, VIDEOS


if __name__ == "__main__":
    # todo сначала загрузим видео и достанем всю информацию, которую можно вытащить

    TRANSCRIPT, LINK = VIDEOS[4].popitem()  # todo change the way of storing it
    TRANSCRIPT = TRANSCRIPT + '.csv'  # todo pizdets

    youtube = YouTubeService()
    video_id = youtube.extract_video_id(LINK)
    channel_id = youtube.get_channel_id(video_id)
    comments = youtube.get_comments(video_id)
    description = youtube.get_channel_description(channel_id)
    video_descriptions = youtube.get_channel_videos_descriptions(channel_id)

    youtube.download_video(video_id=video_id)

    # todo работа с текстами
    processor = DataProcessor(TEXTS_PATH, TRANSCRIPT)
    df = processor.prepare_data(video_id)

    # todo Обогатим df с Ютуба
    comments_df = pd.DataFrame(comments)
    videos_df = pd.DataFrame(video_descriptions)

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
