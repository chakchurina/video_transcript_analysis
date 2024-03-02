import os
import logging
from pandas import DataFrame
from typing import Tuple, List
from moviepy.editor import VideoFileClip, concatenate_videoclips

from config.config import VIDEOS_PATH, SHORTS_PATH


class VideoEditor:
    @staticmethod
    def create_video(df: DataFrame, sentence_numbers: Tuple[int], video_id: str, index: int) -> None:
        # todo add docstring
        source_path: str = os.path.join(VIDEOS_PATH, f'{video_id}.mp4')
        result_path: str = os.path.join(SHORTS_PATH, f'{video_id}_{index}.mp4')

        video: VideoFileClip = VideoFileClip(source_path)
        clips: List[VideoFileClip] = []

        for number in sentence_numbers:
            start_time: float = df.loc[df.index == number, 'start_time'].values[0]
            end_time: float = df.loc[df.index == number, 'end_time'].values[0]
            clip: VideoFileClip = video.subclip(start_time, end_time)
            clips.append(clip)

        final_clip: VideoFileClip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(result_path, codec="libx264", fps=24, audio_codec="aac")

        logging.info(f"Video saved to {result_path}")
