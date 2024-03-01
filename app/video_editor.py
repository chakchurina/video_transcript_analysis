import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

from config.config import VIDEOS_PATH, SHORTS_PATH


class VideoEditor:
    def __init__(self, video_id):
        self.source_path = os.path.join(VIDEOS_PATH, f'{video_id}.mp4')
        self.result_path = os.path.join(SHORTS_PATH, f'{video_id}.mp4')

    def cut_sentences_from_video(self, df, sentence_numbers):
        video = VideoFileClip(self.source_path)
        clips = []

        for number in sentence_numbers:
            start_time = df.loc[df.index == number, 'start_time'].values[0]
            end_time = df.loc[df.index == number, 'end_time'].values[0]
            clip = video.subclip(start_time, end_time)
            clips.append(clip)

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(self.result_path, codec="libx264", fps=24, audio_codec="aac")
