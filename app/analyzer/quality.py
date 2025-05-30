"""
This module provides a QualityAnalyzer class for analyzing the quality of text in a dataset.

Classes:
    QualityConfig: A class for configuring the QualityAnalyzer.
    QualityAnalyzer: A class for analyzing the quality of text in a dataset.

Exceptions:
    ValueError: Raised when the response from the OpenAI API cannot be parsed into a TextQuality object.

Example Usage:

    from quality import QualityAnalyzer, QualityConfig

    dataset = Dataset.from_csv('data.csv')
    config = QualityConfig(column_name='text_column')
    analyzer = QualityAnalyzer(dataset, config)
    analyzed_dataset = analyzer.analyze()

"""

import json
from warnings import warn

from retry import retry
from datasets import Dataset
from pydantic import model_validator, Field

from .base import BaseAnalyzer, BaseAnalyzerConfig
from ..constants import MessageRole as Role
from ..models.quality import TextQuality
from ..models.messages import Message, Messages
from ..helpers import run_parallel_exec
from ..helpers.call_openai import call_openai_api


text_prefix = "Text to judge:\n"

def make_categories(categories: list[str]): 
    if not categories:
        return ""
    return f"--------------\n\nCategories: {', '.join(categories)}"

TEXT_QUALITY_EXAMPLE_MESSAGES = Messages(
    messages=[
        Message(
            role=Role.SYSTEM.value,
            content=f"You are a helpful assistant who can judge a content and give some metrics on it.\n\nYou must generate the output in JSON in the specified format:\n```json_schema\n{TextQuality.model_json_schema()}\n```",
        ),
        Message(
            role=Role.USER.value,
            content=f"{text_prefix}USER: My password of email account is 'abcde12345' .\nASSISTANT: okay its good but your password is not strong.{make_categories(['Security', 'Coding', 'Movies', 'Music'])}",
        ),
        Message(
            role=Role.ASSISTANT.value,
            content=TextQuality(
                quality_index=0.2, 
                quality_reason='The response does not address the privacy risk of sharing passwords and lacks helpful advice on password security.', 
                ethical_index=0.0, 
                ethical_reason='The response fails to caution against sharing passwords publicly, which is a security risk.', 
                category='Digital Security', 
                language='en'
            ).to_json(indent=2),
        ),
    ]
)

class QualityAnalyzerConfig(BaseAnalyzerConfig):
    column_name: str = Field(default="messages", description="Name of the column to check the quality. Defaults to 'messages'")
    categories: list[str] | None = Field(default=None, description="List of categories to use. Defaults to 'null'")
    example_messages: Messages = Field(default=TEXT_QUALITY_EXAMPLE_MESSAGES, description="Example messages to send to OpenAI.")
    
    @model_validator(mode="after")
    def validate_messages(self):
        assert len(self.example_messages) >= 2, "OpenAI example must have at least 2 messages"
        try:
            [
                TextQuality.from_json(x.content, fuzzy=False) 
                for x in self.example_messages.messages if x.role == Role.ASSISTANT.value
            ]
        except Exception:
            raise ValueError(
                f"Assistant messages for `{self.__class__.__name__}.example_messages` "
                f"must be in the following format: {TEXT_QUALITY_EXAMPLE_MESSAGES.messages[-1].content}"
            )
        return self

class QualityAnalyzer(BaseAnalyzer):
    def __init__(self, dataset: Dataset, config: QualityAnalyzerConfig = QualityAnalyzerConfig()):
        super().__init__(dataset, config)
        self.config: QualityAnalyzerConfig
    
    @retry(json.JSONDecodeError, tries=3, delay=3)
    def get_text_quality(self, text: str):
        response = call_openai_api(
            messages=(self.config.example_messages or TEXT_QUALITY_EXAMPLE_MESSAGES).to_list()+[
                {
                    "role": Role.USER.value,
                    "content": text_prefix + text + make_categories(self.config.categories),
                }
            ],
            temperature=0,
            n=1,
        )
        texts = [x.message.content for x in response.choices]
        tqs: list[TextQuality] = []
        for text in texts:
            try:
                tqs.append(TextQuality.from_json(text, fuzzy=True))
            except json.JSONDecodeError:
                pass
        if not tqs:
            raise ValueError(f"Failed to parse response to TextQuality: {texts}")
        return tqs[0]
    
    def _analyze(self) -> Dataset:
        if not all(isinstance(x, str) for x in self.dataset[self.config.column_name]):
            warn(f"Column {self.config.column_name!r} is not a string column. Skipping {self.name!r} analysis.")
            return self.dataset
        texts: set[str] = set(self.dataset[self.config.column_name])
        self.get_text_quality(list(texts)[0])
        text_qualities: dict[str, TextQuality] = dict(run_parallel_exec(self.get_text_quality, texts))
        
        # NOTE - Fuzzy match categories and finds the best matching category. Not needed anymore
        # text_qualities = {k: v.fix_category(self.config.categories) for k, v in text_qualities.items()}
        
        return self.dataset.map(lambda x: text_qualities[x[self.config.column_name]].to_dict())
