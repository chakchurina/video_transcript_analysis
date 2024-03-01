import os

from googleapiclient.discovery import build
from pytube import YouTube

from config.config import YOUTUBE_API_KEY, VIDEOS_PATH


class YouTubeService:
    def __init__(self):
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.video_url = "https://www.youtube.com/watch?v="

    def get_comments(self, video_id):
        comments = []
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

    def get_channel_id(self, video_id):
        request = self.youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = request.execute()

        if 'items' in response and response['items']:
            return response['items'][0]['snippet']['channelId']
        else:
            return None

    def get_channel_description(self, channel_id):
        request = self.youtube.channels().list(
            part='snippet',
            id=channel_id
        )
        response = request.execute()

        if 'items' in response and response['items']:
            return response['items'][0]['snippet']['description']
        else:
            return None

    def get_channel_videos_descriptions(self, channel_id):
        video_descriptions = []
        request = self.youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            order="date"
        )
        response = request.execute()

        for item in response['items']:
            if item['id']['kind'] == "youtube#video":
                video_description = {
                    "title": item['snippet']['title'],
                    "description": item['snippet']['description']
                }
                video_descriptions.append(video_description)

        return video_descriptions

    def download_video(self, video_id, save_path=VIDEOS_PATH):
        filename = video_id + '.mp4'
        file_path = os.path.join(save_path, filename)

        if os.path.exists(file_path):
            print(f'Video {video_id} already exists. Skipping download.')
            return

        video_url = f'{self.video_url}{video_id}'
        yt = YouTube(video_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

        # todo raise exception
        if stream:
            stream.download(output_path=save_path, filename=filename)
            print(f'Video {video_id} has been downloaded successfully.')
        else:
            print('No suitable stream found for downloading.')

    @staticmethod
    def extract_video_id(url):
        # assuming the URL format be like https://www.youtube.com/watch?v=VIDEO_ID
        prefix = "watch?v="
        start_index = url.find(prefix) + len(prefix)
        video_id = url[start_index:]
        return video_id

