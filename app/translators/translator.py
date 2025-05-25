from typing import Any, Callable

from datasets import Dataset

from ..helpers.call_openai import call_openai_api
from ..helpers.utils import recursive_string_operator, run_parallel_exec_but_return_in_order


class Translator:
    """A class for translating strings. Can be used to translate nested objects like lists, dictionaries, Datasets, etc."""
    
    def __init__(self, language: str, predicate: Callable[[str], str|None] = None) -> None:
        """
        Initializes the Translator class.
        
        Parameters:
            language (str): The language to translate to.
            predicate ((str) -> (str | None)): A function that takes in a string and returns a string or None. 
                If a string is returned, that string will be returned. 
                If None is returned, the input string will be translated. Defaults to None.
        
        Returns:
            None
        """
        self.language = language
        self.predicate = predicate

    def _translate_text(self, text: str) -> str:
        response = call_openai_api(
            messages=[
                {"role": "system", "content": "You are a great translator. You should only return the translated text and nothing else. You can skip those words that don't have meaning in the translated language."},
                {"role": "user", "content": "'What are the latest developments in quantum computing as of 2023?' to Armenian"},
                {"role": "assistant", "content": "2023 \u0569\u057e\u0561\u056f\u0561\u0576\u056b\u0576 \u056b\u0576\u0579\u057a\u056b\u057d\u056b\u0576 \u0565\u0576 \u057e\u0565\u0580\u057b\u056b\u0576 \u0566\u0561\u0580\u0563\u0561\u0581\u0578\u0582\u0574\u0576\u0565\u0580\u0568 \u0584\u057e\u0561\u0576\u057f\u0561\u0575\u056b\u0576 \u0570\u0561\u0577\u057e\u0561\u0580\u056f\u0574\u0561\u0576 \u0574\u0565\u057b\u0589"},
                {"role": "user", "content": "'This is a text to translate' to Japanese"},
                {"role": "assistant", "content": "これは翻訳するための「text」です"},
                {"role": "user", "content": "'Summarize the current progress on the Mars exploration missions in 2023 based on the latest news, articles, and space agency updates. Contexts: {{step_1.function.function_1}}; {{step_2.extract.extract_2}}; {{step_3.function.function_3}}.' to Bengali"},
                {"role": "assistant", "content": "২০২৩ সালের মঙ্গল অনুসন্ধান মিশনের বর্তমান অগ্রগতি সম্পর্কে সারসংক্ষেপ দিন, সর্বশেষ খবর, নিবন্ধ এবং মহাকাশ সংস্থার আপডেট অনুযায়ী। প্রেক্ষাপট: {{step_1.function.function_1}}; {{step_2.extract.extract_2}}; {{step_3.function.function_3}}।"},
                {"role": "user", "content": "'Markdown' to Dutch"},
                {"role": "assistant", "content": "Markdown"},
                {"role": "user", "content": "'What are the latest developments in quantum computing as of 2023?' to Sanskrit"},
                {"role": "assistant", "content": "\u0968\u0966\u0968\u0969 \u0924\u092e\u0947 \u0935\u0930\u094d\u0937\u0947 \u0915\u094d\u0935\u093e\u0923\u094d\u091f\u092e\u094d \u0938\u0919\u094d\u0917\u0923\u0915\u0924\u093e\u092f\u093e\u0903 \u0928\u0935\u0940\u0928\u0924\u092e\u093e\u0903 \u0935\u093f\u0915\u093e\u0938\u093e\u0903 \u0915\u0947 \u0938\u0928\u094d\u0924\u093f?"},
                {"role": "user", "content": "'PubMed' to Hindi"},
                {"role": "assistant", "content": "PubMed"},
                {"role": "user", "content": "'Alzheimer' to Hindi"},
                {"role": "assistant", "content": "अल्जाइमर"},
                {"role": "user", "content": "'Summarize the current progress on the Mars exploration missions in 2023 based on the latest news, articles, and space agency updates. Contexts: {{step_1.function.function_1}}; {{step_2.function.function_2}}; {{step_3.function.function_3}}.' to Bengali"},
                {"role": "assistant", "content": "২০২৩ সালে মঙ্গল অভিযানের বর্তমান অগ্রগতি সারসংক্ষেপ করুন, সর্বশেষ খবর, প্রবন্ধ এবং মহাকাশ সংস্থার আপডেটের ভিত্তিতে। প্রেক্ষাপটগুলি: {{step_1.function.function_1}}; {{step_2.function.function_2}}; {{step_3.function.function_3}}।"},
                {"role": "user", "content": f"{text!r} to {self.language}"}
            ]
        )
        if not hasattr(response, "choices"):
            return response
        return response.choices[0].message.content
        
    def _operator(self, text: str):
        if callable(self.predicate):
            r = self.predicate(text)
            if r is not None:
                return r
        return self._translate_text(text)
        
    def _translate_any(self, data, skip_keys: list[str] = [], workers: int = 4) -> Any:
        return recursive_string_operator(
            data=data, fn=self._operator, skip_keys=skip_keys, max_workers=workers
        )
        
    def translate(self, data, skip_keys: list[str] = [], workers: int = 4) -> Any:
        return self._translate_any(data, skip_keys, workers)
    
    def translate_many(self, data_list: list[Any], skip_keys: list[str] = [], workers: int = 4) -> Any:
        internal_workers = max(abs(5 - min(5, workers)), 1)
        return run_parallel_exec_but_return_in_order(
            self.translate, data_list, skip_keys, internal_workers, max_workers=workers
        )
        
    def translate_dataset(self, dataset: Dataset, column_name: str, skip_keys: list[str] = [], workers: int = 4) -> Dataset:
        if column_name not in dataset.column_names:
            raise ValueError(f"Column '{column_name}' not found in dataset.")
        data = self.translate_many(dataset[column_name], skip_keys=skip_keys, workers=workers)
        return dataset.add_column(f"{column_name}_{self.language}", data)
    
    def __call__(self, data, skip_keys: list[str] = [], workers: int = 4) -> Any:
        return self._translate_any(data, skip_keys, workers)
