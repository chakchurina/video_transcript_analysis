import os
from utils import load_transcript_data, get_transcript_as_list, get_transcript_as_str
import random


def predict_topic(list_of_transcripts):
    """ Fake topic predictor """
    predictions = []
    for idx, line in enumerate(list_of_transcripts):
        current_pred = random.choices(["Greeting", "Main Topic 1", "Main Topic 2"], weights=[0.2,0.4,0.4],k=1)
        predictions.append([idx,current_pred])

    return predictions


if __name__ == "__main__":
    file1 = "2024 Rolls-Royce Spectre Review.csv"
    file2 = "Apple Vision Pro Impressions.csv"
    file3 = "George Hotz.csv"
    file4 = "The END of Sam Bankman Fried.csv"
    file5 = "Why is LinkedIn so weird.csv"

    folder = "transcripts"

    data = load_transcript_data(os.path.join(folder,file5))
    list_of_transcripts = get_transcript_as_list(data[:10], include_index=True)
    transcript = get_transcript_as_str(list_of_transcripts)

    predictions = predict_topic(list_of_transcripts)
    
    print(transcript)
    print(predictions)