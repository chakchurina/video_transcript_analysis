from app.analytics.preprocessor import DataProcessor
from app.analytics.summarizer import TextSummarizer
from app.analytics.insight_extractor import InsightExtractor
from app.analytics.segmenter import TextSegmenter
from app.services.llm_service import LLM
from app.services.youtube_service import YouTubeService
from app.video_editor import VideoEditor

from config.config import TEXTS_PATH, VIDEOS


if __name__ == "__main__":

    # todo change the way of storing it
    TRANSCRIPT, LINK = VIDEOS[4]
    TRANSCRIPT = TRANSCRIPT + '.csv'  # todo cringe

    # get video and comments from YouTube
    youtube = YouTubeService()
    video_id = youtube.extract_video_id(LINK)
    # youtube.download_video(video_id=video_id)
    # comments = youtube.get_comments(video_id)

    # preprocess transcript and add auxiliary columns
    preprocessor = DataProcessor(TEXTS_PATH, TRANSCRIPT, video_id)
    df = preprocessor.create_dataframe()

    summarizer = TextSummarizer()
    # get video keywords
    theme_keywords = summarizer.get_keywords(df)
    # get sentences that best describe the whole video
    summary = summarizer.summarize(df, 3)

    extractor = InsightExtractor(df)
    segmenter = TextSegmenter(df)
    llm = LLM()

    emotionals = extractor.emotional_messages()
    questions = extractor.questions()
    intros = extractor.intros()

    texts = {}
    for sentence_index in set(emotionals + questions + intros):
        # Get 0 paragraphs before, 2 after and retrieve all relevant
        consecutive = segmenter.get_consecutive(sentence_index, 0, 2)
        closest = segmenter.get_n_closest(sentence_index, n=2)
        context = list(sorted(set(consecutive + closest)))

        context_string = "\n".join([f"{index}: {df.loc[index, 'sentence']}" for index in context])

        generated = llm.generate(sentence_index, context_string, theme_keywords)

        text = ' '.join(df.loc[generated, 'sentence'])
        texts.update({tuple(generated): text})

    # todo
    scripts = []
    for i, (_, text) in enumerate(texts.items()):
        scripts.append(f"{i}\n{text}\n\n")

    selected = llm.validate(texts=scripts, number=5)

    for entry in selected:
        indexes_set = list(texts.keys())[entry]
        text = ' '.join(df.loc[list(indexes_set), 'sentence'])
        print(text)

        cutter = VideoEditor(video_id)
        cutter.cut_sentences_from_video(df, selected)
