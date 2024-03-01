import os
import logging
from typing import List, Dict, Optional

from googleapiclient.discovery import build
from pytube import YouTube

from config.config import YOUTUBE_API_KEY, VIDEOS_PATH


class YouTubeService:
    """Service class to interact with YouTube API and download videos."""

    def __init__(self) -> None:
        """Initializes the YouTube API client and sets video URL prefix."""
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.video_prefix = "https://www.youtube.com/watch?v="

    def get_comments(self, video_id: str) -> List[Dict[str, any]]:
        """
        Fetches comments for a given YouTube video.

        Args:
            video_id (str): Unique identifier of the YouTube video.

        Returns:
            List[Dict[str, any]]: Sorted list of comments by likes in descending order.
        """
        comments: List[Dict[str, any]] = []
        request = self.youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100,
        )
        response = request.execute()

        for item in response['items']:
            comment = {
                "id": item['snippet']['topLevelComment']['id'],
                "text": item['snippet']['topLevelComment']['snippet']['textDisplay'],
                "likes": item['snippet']['topLevelComment']['snippet']['likeCount']
            }
            comments.append(comment)

        return sorted(comments, key=lambda item: item["likes"], reverse=True)

    def get_channel_id(self, video_id: str) -> Optional[str]:
        """
        Retrieves the channel ID of a given YouTube video.

        Args:
            video_id (str): Unique identifier of the YouTube video.

        Returns:
            Optional[str]: Channel ID if found, else None.
        """
        request = self.youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = request.execute()

        if 'items' in response and response['items']:
            return response['items'][0]['snippet']['channelId']
        else:
            return None

    def download_video(self, video_id: str, save_path: str = VIDEOS_PATH) -> None:
        """
        Downloads a YouTube video to a specified path.

        Args:
            video_id (str): Unique identifier of the YouTube video.
            save_path (str): Directory path to save the downloaded video.
        """
        filename = f'{video_id}.mp4'
        file_path = os.path.join(save_path, filename)

        if os.path.exists(file_path):
            logging.info(f'Video {video_id} already exists. Skipping download.')
            return

        video_url = f'{self.video_prefix}{video_id}'
        yt = YouTube(video_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

        if stream:
            stream.download(output_path=save_path, filename=filename)
            logging.info(f'Video {video_id} has been downloaded successfully.')
        else:
            logging.warning('No suitable stream found for downloading.')

    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Extracts the video ID from a YouTube URL.

        Args:
            url (str): Full URL of the YouTube video.

        Returns:
            str: Extracted video ID.
        """
        prefix = "watch?v="
        start_index = url.find(prefix) + len(prefix)
        video_id = url[start_index:]
        return video_id
