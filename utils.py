import csv


def load_transcript_data(file_path):
    """ Loads file. returns list of rows(type: list)."""
    with open(file_path) as csvfile:
        results = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        data = [row for row in results]
    return data


def get_transcript_as_list(csv_data, include_index=False, remove_first_row=True):
    """extracts text data as list, with index prepended (as string)."""
    if remove_first_row:
        starting_idx = 1
    else:
        starting_idx = 0
    if include_index:
        text_data = [str(idx)+" "+row[1] for idx,row in enumerate(csv_data[starting_idx:])]
    else:
        text_data = [row[1] for idx,row in enumerate(csv_data[starting_idx:])]
    
    return text_data


def get_transcript_as_str(list_of_transcripts):
    """Creates a transcript file."""
    transcript = "\n".join(list_of_transcripts)
    return transcript