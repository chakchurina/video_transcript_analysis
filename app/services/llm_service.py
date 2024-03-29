import logging
from typing import Tuple, List, Any
from pandas import DataFrame

from app.analytics.base_processor import BaseTextProcessor
from config.config import GPT_MODEL
from config.prompts import get_sentences_prompt_template, verification_prompt_template


class LLM(BaseTextProcessor):
    """
    Language Model Processor class for generating and validating text.
    """
    def __init__(self) -> None:
        """
        Initializes the class with a specific language model.
        """
        self.model: str = GPT_MODEL

    def generate(self, df: DataFrame, sentence_number: int, context: List[int], keywords: List[str]) -> Tuple[int, ...]:
        """
        Generates a list of sentence numbers based on the sentence number and its context.

        Args:
            df: DataFrame containing the data.
            sentence_number: The central sentence number to focus on.
            context: A list of indices representing the context sentences.
            keywords: A list of keywords to guide the generation.

        Returns:
            A list of integers representing generated sentence numbers.
        """
        context_string: str = "\n".join([f"{index}: {df.loc[index, 'sentence']}" for index in context])
        theme: str = ", ".join(keywords)
        smallest: int = 6
        largest: int = 10

        prompt_text: str = get_sentences_prompt_template.format(
            smallest=smallest,
            largest=largest,
            theme=theme,
            central=sentence_number,
            transcript=context_string
        )

        try:
            response: Any = self.client.chat.completions.create(
                model=self.model,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are a professional copywriter and text editor"},
                    {"role": "user", "content": prompt_text}
                ]
            )

            text: str = response.choices[0].message.content
            sentences: Tuple[int] = tuple(map(int, text.split(",")))
        except ValueError as e:
            logging.error(f"Error processing model output: {e}")
            return ()
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return ()

        if smallest <= len(sentences) <= largest:
            return sentences
        else:
            return ()

    def validate(self, scripts: str, largest: int) -> List[int]:
        """
        Validates a given list of scripts against a specified number using the model.

        Args:
            scripts: A list of scripts to be validated.
            largest: The number against which the validation is to be performed.

        Returns:
            A list of integers representing validation results.
        """
        prompt_text: str = verification_prompt_template.format(scripts=scripts, n=largest)

        try:
            response: Any = self.client.chat.completions.create(
                model=self.model,
                temperature=1,
                messages=[
                    {"role": "system", "content": "You are a professional copywriter and text editor"},
                    {"role": "user", "content": prompt_text}
                ]
            )

            text: str = response.choices[0].message.content
            numbers: List[int] = list(map(int, text.split(",")))
        except ValueError as e:
            logging.error(f"Error processing model output: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return []

        if len(numbers) <= largest:
            return numbers
        else:
            return []
