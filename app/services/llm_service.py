from app.analytics.base_processor import BaseTextProcessor
from config.config import GPT_MODEL
from config.promts import get_sentences_prompt_template, verification_prompt_template


class LLM(BaseTextProcessor):
    def __init__(self):
        self.model = GPT_MODEL

    def generate(self, sentence_number, transcript, keywords):

        theme = ", ".join(keywords)
        smallest = 6
        largest = 12

        print(f"Prompting with {smallest}, {largest}, sentence {sentence_number}")

        prompt_text = get_sentences_prompt_template.format(smallest=smallest,
                                                           largest=largest,
                                                           theme=theme,
                                                           central=sentence_number,
                                                           keywords=keywords,
                                                           transcript=transcript)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a professional copywriter and text editor"},
                {"role": "user", "content": prompt_text}
            ])

        text = response.choices[0].message.content
        sentences = list(map(int, text.split(",")))  # todo add guardrails

        return sentences

    def validate(self, number, texts):

        prompt_text = verification_prompt_template.format(scripts=texts,
                                                          n=number)

        response = self.client.chat.completions.create(
            model=GPT_MODEL,
            temperature=1,
            messages=[
                {"role": "system", "content": "You are a professional copywriter and text editor"},
                {"role": "user", "content": prompt_text}
            ])

        text = response.choices[0].message.content
        numbers = list(map(int, text.split(",")))  # todo add guardrails

        return numbers
