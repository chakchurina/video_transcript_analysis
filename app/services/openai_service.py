from openai import OpenAI

from config.config import OPENAI_API_KEY, GPT_MODEL
from config.promts import get_sentences_prompt_template


@staticmethod
def calculate_embeddings(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model=EMBEDDINGS_MODEL)
    return response.data[0].embedding


def prompt_gpt(sentence_number, transcript, keywords):

    theme = ", ".join(keywords)
    smallest = 7
    largest = 15

    print(f"Prompting with {smallest}, {largest}, {theme}, {sentence_number}")

    prompt_text = get_sentences_prompt_template.format(smallest=smallest,
                                                       largest=largest,
                                                       theme=theme,
                                                       central=sentence_number,
                                                       keywords=keywords,
                                                       transcript=transcript)

    client = OpenAI(api_key=OPENAI_API_KEY)  # todo move to connector class

    response = client.chat.completions.create(
        model=GPT_MODEL,  # todo move to config
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a professional copywriter and text editor"},
            {"role": "user", "content": prompt_text}
        ])

    text = response.choices[0].message.content
    sentences = list(map(int, text.split(",")))  # todo add guardrails

    return sentences
