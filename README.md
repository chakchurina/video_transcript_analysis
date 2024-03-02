# Rask AI take-home explorative assignment

## Task Overview

Hi, Brask Team!

Thank you for the opportunity and the test assignment! It was genuinely fun and engaging.

You requested an explorative analysis, but as a practice-oriented person, I discovered during my research that video segmentation was possible, so I went ahead and implemented it.

In this repository, you'll find a small service that executes such segmentation. The main idea revolves around:
- Attempting to identify the most vivid parts of the text (using sentiment analysis, topic modeling, etc.),
- Finding the context of these highlights through cosine distance,
- And prompting Chat GPT to form short stories from these sentences.

You'll find the prompts and all the modules needed for this analysis inside. I attempted to validate the results using ChatGPT.

**What's Missing**
- Lack of precise time-codes from phrases: I estimated based on sentence durations, but the video splice was not high quality. I chose not to calculate directly from the video to save time.
- Absence of Docker, and all other development attributes, but I decided it wasn't the focus of the assignment and didn't spend time on them. Mainly, I wanted to achieve a practical result.

**Ideas I Had But Didn't Implement**

- I tried enriching the dataset with YouTube comments — they could help identify frequently commented spots. YouTube doesn't provide hotspots, but extracting them, too, seems like a good idea.
- I would also try to operate not on individual sentences, but rather on pairs, triplets, or entire paragraphs.
- Choosing themes for shorts based on channel descriptions or YouTube topics might improve views for a specific blogger.
- If video processing speed isn't an issue, exploring the video with Computer Vision to alternate shots and make the cuts neater could be tried. 
- Based on sentiment analysis, I would create a small meme library — they could be added to the video at moments with high sentiment analysis scores.
![grumpy.png](https://www.palmbeachpost.com/gcdn/authoring/2013/06/02/NPPP/ghows-LK-b5f8745d-d134-4a5c-8d6c-cf570b96573a-2f43bf6d.jpeg?width=660&height=372&fit=crop&format=pjpg&auto=webp)

The repository includes a notebook with my experiments — I tried various approaches before packaging my solutions into the service. You can browse through it, but don't consider it a solution; it's more like my jotter, which demonstrates my thought process.

I want to say thanks again for such a cool task! I enjoyed working on it and hope you'll like my work too!

## Features

- **Sentiment Analysis**: Evaluate the sentiment of text data, categorizing it into positive, negative, or neutral sentiments.
- **Text Summarization**: Extract key sentences from a large body of text to create concise summaries.
- **Keyword Extraction**: Identify and extract the most relevant keywords from text data.
- **Text Segmentation**: Segment text into meaningful blocks or paragraphs based on semantic similarities.

## How to Run

Make sure you have Python 3.8+ and pip installed. Then, follow these steps:

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Set Up Configuration**:
   - Update `config/config.py` with the necessary API keys.

4. **Select a file for processing**:
   - Change the `FILE_NUMBER` variable in `config/config.py`

6. **Run the Application**:
   - Execute the main script to start processing your data.
     ```
     python app.py
     ```
     
Look for resulting videos at `data/shorts/` and find the texts at `data/`.

## Project Structure

- **`app/`**: Main application directory.
  - **`analytics/`**: Contains modules for data processing, sentiment analysis, text summarization, and more.
    - `base_processor.py`: Base class for text processing.
    - `sentiment_analyzer.py`: Module for sentiment analysis.
    - `text_summarizer.py`: Module for generating text summaries.
    - `insight_extractor.py`: Extracts insights from text data.
    - `text_segmenter.py`: Segments text into meaningful blocks.
  - **`services/`**: External services integration (e.g., YouTube API).
- **`config/`**: Configuration files and constants.
  - `config.py`: Central configuration file. Left it as .py file for the sake of simplicity 
- **`data/`**: Directory for storing input data and generated outputs.
- **`requirements.txt`**: Project dependencies.

## Enjoy and Have fun.

(C) Brask Inc. All rights reserved.