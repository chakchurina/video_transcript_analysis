import unittest
from unittest.mock import patch, MagicMock
from app.services.youtube_service import YouTubeService


class TestYouTubeService(unittest.TestCase):
    @patch('app.services.youtube_service.build')
    def setUp(self, mock_build):
        self.youtube_service = YouTubeService()

    def test_extract_video_id(self):
        url = "https://www.youtube.com/watch?v=abcdefg"
        expected_video_id = "abcdefg"
        self.assertEqual(YouTubeService.extract_video_id(url), expected_video_id)

    @patch('app.services.youtube_service.YouTube')
    def test_download_video(self, mock_youtube):
        mock_stream = MagicMock()
        mock_stream.download = MagicMock(return_value=None)
        mock_streams = MagicMock()
        mock_streams.filter.return_value = mock_streams
        mock_streams.order_by.return_value = mock_streams
        mock_streams.desc.return_value = mock_streams
        mock_streams.first.return_value = mock_stream
        mock_youtube.return_value.streams = mock_streams

        video_id = "abcdefg"
        save_path = "path/to/save"
        self.youtube_service.download_video(video_id, save_path)
        mock_stream.download.assert_called_once_with(output_path=save_path, filename=f"{video_id}.mp4")

    @patch('app.services.youtube_service.build')
    def test_get_comments(self, mock_build):
        mock_execute = MagicMock()
        mock_execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "id": "1",
                            "snippet": {
                                "textDisplay": "comment1",
                                "likeCount": 10
                            }
                        }
                    }
                },
                {
                    "snippet": {
                        "topLevelComment": {
                            "id": "2",
                            "snippet": {
                                "textDisplay": "comment2",
                                "likeCount": 20
                            }
                        }
                    }
                }
            ]
        }
        mock_request = MagicMock()
        mock_request.execute = mock_execute
        mock_comment_threads = MagicMock()
        mock_comment_threads.list.return_value = mock_request
        mock_build.return_value.commentThreads.return_value = mock_comment_threads

        video_id = "abcdefg"
        expected_comments = [
            {"id": "2", "text": "comment2", "likes": 20},
            {"id": "1", "text": "comment1", "likes": 10}
        ]
        self.assertEqual(YouTubeService().get_comments(video_id), expected_comments)

    @patch('app.services.youtube_service.build')
    def test_get_channel_id(self, mock_build):
        mock_execute = MagicMock()
        mock_execute.return_value = {
            "items": [{"snippet": {"channelId": "channel123"}}]
        }
        mock_request = MagicMock()
        mock_request.execute = mock_execute
        mock_videos = MagicMock()
        mock_videos.list.return_value = mock_request
        mock_build.return_value.videos.return_value = mock_videos

        video_id = "abcdefg"
        self.assertEqual(YouTubeService().get_channel_id(video_id), "channel123")


if __name__ == '__main__':
    unittest.main()
