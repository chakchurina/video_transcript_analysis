import logging
from typing import List, Dict, Tuple

from pandas import DataFrame
from app.analytics.preprocessor import DataProcessor
from app.analytics.summarizer import TextSummarizer
from app.analytics.insight_extractor import InsightExtractor
from app.analytics.segmenter import TextSegmenter
from app.services.llm_service import LLM
from app.services.youtube_service import YouTubeService
from app.video_editor import VideoEditor

from config.config import TEXTS_PATH, VIDEOS, FILE_NUMBER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    transcript, link = VIDEOS[FILE_NUMBER]

    youtube = YouTubeService()
    video_id: str = youtube.extract_video_id(link)
    youtube.download_video(video_id=video_id)

    preprocessor = DataProcessor(TEXTS_PATH, transcript, video_id)
    df: DataFrame = preprocessor.create_dataframe()

    summarizer = TextSummarizer()
    theme_keywords: List[str] = summarizer.get_keywords(df)
    summary: List[int] = summarizer.summarize(df, 3)

    extractor = InsightExtractor(df)
    segmenter = TextSegmenter(df)
    llm = LLM()

    highlights: List[int] = extractor.get_highlights()

    texts: Dict[Tuple[int], str] = {}
    for sentence_index in highlights + summary:
        consecutive: List[int] = segmenter.get_consecutive(sentence_index, 0, 2)
        closest: List[int] = segmenter.get_n_closest(sentence_index, n=2)
        context: List[int] = sorted(set(consecutive + closest))

        generated: List[int] = llm.generate(df, sentence_index, context, theme_keywords)

        text: str = ' '.join(df.loc[generated, 'sentence'])
        texts[tuple(generated)] = text

    # todo: better move it to a function
    scripts: List[str] = [f"{i}\n{text}\n\n" for i, (_, text) in enumerate(texts.items())]

    selected: List[int] = llm.validate(scripts=scripts, number=5)

    for i, entry in enumerate(selected):
        indexes_set: Tuple[int] = list(texts.keys())[entry]
        text: str = ' '.join(df.loc[list(indexes_set), 'sentence'])

        logging.info(f"Cutting script: {text}")
        VideoEditor.create_video(df, selected, video_id, i)


if __name__ == "__main__":
    main()
