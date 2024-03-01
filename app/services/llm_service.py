import logging
from typing import List, Any
from pandas import DataFrame
from app.analytics.base_processor import BaseTextProcessor
from config.config import GPT_MODEL
from config.prompts import get_sentences_prompt_template, verification_prompt_template


class LLM(BaseTextProcessor):
    def __init__(self) -> None:
        self.model: str = GPT_MODEL

    def generate(self, df: DataFrame, sentence_number: int, context: List[int], keywords: List[str]) -> List[int]:
        context_string: str = "\n".join([f"{index}: {df.loc[index, 'sentence']}" for index in context])
        theme: str = ", ".join(keywords)
        smallest: int = 6
        largest: int = 12

        logging.info(f"Prompting with {smallest}, {largest}, sentence {sentence_number}")

        prompt_text: str = get_sentences_prompt_template.format(
            smallest=smallest,
            largest=largest,
            theme=theme,
            central=sentence_number,
            keywords=keywords,
            transcript=context_string
        )

        response: Any = self.client.chat.completions.create(
            model=self.model,
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a professional copywriter and text editor"},
                {"role": "user", "content": prompt_text}
            ]
        )

        text: str = response.choices[0].message.content
        sentences: List[int] = list(map(int, text.split(",")))  # Consider adding validation for the output

        return sentences

    def validate(self, scripts: List[str], number: int) -> List[int]:
        prompt_text: str = verification_prompt_template.format(scripts=scripts, n=number)

        response: Any = self.client.chat.completions.create(
            model=self.model,
            temperature=1,
            messages=[
                {"role": "system", "content": "You are a professional copywriter and text editor"},
                {"role": "user", "content": prompt_text}
            ]
        )

        text: str = response.choices[0].message.content
        numbers: List[int] = list(map(int, text.split(",")))  # Consider adding validation for the output

        return numbers
