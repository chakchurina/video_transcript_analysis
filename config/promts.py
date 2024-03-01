get_sentences_prompt_template = """
Input:
You have part of a podcast transcript with sentence numbers in a format <number: sentence>. 

Task:
Select sentences from the input text, so that they tell a coherent story.

Additionally:
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

verification_prompt_template = """
Input:
You have a sequence scripts for short videos in a format number:text. 

Task:
Select from the provided fragments {n} distinct that read as coherent and logical texts. 
Give preference to fragments that will be as different from each other as possible.

Result: 
Return comma-separated numbers of selected scripts.

Example of the result: 
1, 3, 4, 15, 21

Scripts:
{scripts}

Restrictions:
- output only {n} comma-separated numbers.
"""