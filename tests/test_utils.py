from utils import load_transcript_data, get_transcript_as_list, get_transcript_as_str


if __name__ == "__main__":
    file_data = load_transcript_data("test.csv")
    assert( type(file_data)==list )
    assert( len(file_data)==5 )
    assert( type(file_data[0])==list )
    assert( len(file_data[0])==3 )

    transcript_data = get_transcript_as_list(file_data, include_index=True, remove_first_row=True)
    assert( type(transcript_data)==list )
    assert( len(transcript_data)==4 )
    assert( type(transcript_data[0])==str )

    transcript = get_transcript_as_str(transcript_data)
    assert( type(transcript)==str )

    print("Tests Succeeded")