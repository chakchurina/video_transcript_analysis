import os
import logging
from typing import List, Dict, Tuple
from pandas import DataFrame

from app.analytics.segmenter import TextSegmenter
from app.analytics.preprocessor import DataProcessor
from app.insight_extractor import InsightExtractor
from app.services.llm_service import LLM
from app.services.youtube_service import YouTubeService
from app.video_editor import VideoEditor

from config.config import TEXTS_PATH, VIDEOS, FILE_NUMBER, RESULT_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VideoAnalysisPipeline:
    """
    This class encapsulates the entire process of video analysis.
    """

    def __init__(self, file_number: int):
        self.file_number: int = file_number
        self.transcript, self.link = VIDEOS[file_number]

        self.video_id: str = ''
        self.df: DataFrame = DataFrame()
        self.scripts: Dict[Tuple[int, ...], str] = {}

    def run(self) -> None:
        """
        Executes the video analysis pipeline.
        """
        self.download_video()
        self.process_transcript()
        self.analyze_content()
        self.save_texts()
        self.edit_videos()

    def download_video(self) -> None:
        """
        Downloads the video using YouTubeService.
        """
        youtube: YouTubeService = YouTubeService()
        self.video_id: str = youtube.extract_video_id(self.link)
        youtube.download_video(video_id=self.video_id)

    def process_transcript(self) -> None:
        """
        Processes the video transcript into a DataFrame.
        """
        preprocessor: DataProcessor = DataProcessor(TEXTS_PATH, self.transcript, self.video_id)
        self.df: DataFrame = preprocessor.create_dataframe()

    def analyze_content(self) -> None:
        """
        Analyzes the content to identify sentences for editing videos.
        """
        extractor: InsightExtractor = InsightExtractor(self.df)
        segmenter: TextSegmenter = TextSegmenter(self.df)
        llm: LLM = LLM()

        keywords: List[str]
        summary: List[int]
        keywords, summary = extractor.get_summary(3)
        highlights: List[int] = extractor.get_highlights(10)

        raws: Dict[Tuple[int, ...], str] = {}
        for sentence_index in highlights + summary:
            consecutive: List[int] = segmenter.get_consecutive(sentence_index, 0, 2)
            closest: List[int] = segmenter.get_n_closest(sentence_index, n=2)
            context: List[int] = sorted(set(consecutive + closest))

            generated: Tuple[int] = llm.generate(self.df, sentence_index, context, keywords)

            text: str = ' '.join(self.df.loc[list(generated), 'sentence'])

            if generated and generated not in raws:
                raws[tuple(generated)] = text

        texts = "\n\n".join(f"{i}:\n{text}" for i, text in enumerate(raws.values()))
        selected: List[int] = llm.validate(scripts=texts, largest=5)

        for i, entry in enumerate(selected):
            indices: Tuple[int] = list(raws.keys())[entry]
            text: str = ' '.join(self.df.loc[list(indices), 'sentence'])
            self.scripts.update({indices: text})

    def edit_videos(self) -> None:
        """
        Creates short videos based on the selected sentences.
        """
        for i, (indices, text) in enumerate(self.scripts.items()):
            logging.info(f"Cutting script: {text}")
            VideoEditor.create_video(self.df, indices, self.video_id, i)

    def save_texts(self) -> None:
        """
        Saves the generated text to a file for further reference.
        """
        for _, text in self.scripts.items():
            with open(os.path.join(RESULT_PATH, f"{self.video_id}.txt"), 'a') as file:
                file.write(text + '\n')


if __name__ == "__main__":
    pipeline: VideoAnalysisPipeline = VideoAnalysisPipeline(FILE_NUMBER)
    pipeline.run()
