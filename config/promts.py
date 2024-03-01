get_sentences_prompt_template = """
Input:
You have part of a podcast transcript with sentence numbers in a format <number: sentence>. 

Task:
Pick sentences from the input text, so that they tell a coherent story. 
Think about placing sentence that contain {keywords} first. 
Try to include sentence number {central} in the sequence, if it fits.
Consider adding sentence that has conclusion, generalisation or prognosis, at the end of sequence.

Result: 
Return comma-separated numbers of the sentences that form a readable text.

Example of the result: 
122, 124, 125, 127, 129, 130, 140, 142

Transcript:
{transcript}

Restrictions:
- output only comma-separated numbers,
- there MUST be at least {smallest} sentences in response, but no mora than {largest}.
"""
