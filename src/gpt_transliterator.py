import os
import sys
import openai

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from llm_batch_utils import BaseLLMTransliterator  # noqa: E402


# Transliteration B: Zero-shot transliteration with OpenAI GPT API
class GPTTransliteration(BaseLLMTransliterator):

    def __init__(self, model_name='gpt-4o-mini', request_timeout: float = 60.0):
        super().__init__(provider_label='GPT', min_request_interval=0.0)
        api_key = os.environ.get('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=api_key, timeout=request_timeout, max_retries=1)
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.prompt = self.SINGLE_PROMPT
        self.batch_prompt = self.BATCH_PROMPT

    def _call_single_api(self, prompt_text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            # parameter: temperature is for reproducibility
            temperature=.0,
            timeout=self.request_timeout,
            messages=[{'role': 'user', 'content': prompt_text}]
        )
        return (response.choices[0].message.content or '').strip()

    def _call_batch_api(self, prompt_text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=.0,
            timeout=self.request_timeout,
            response_format={"type": "json_object"},
            messages=[{'role': 'user', 'content': prompt_text}]
        )
        return (response.choices[0].message.content or '').strip()


if __name__ == '__main__':
    transliterator = GPTTransliteration()
    test_words = ["Ки́їв", "Львів", "Ви́года"]

    print("=== 1. Testing Single Transliteration ===")
    single_res = transliterator.transliterate(test_words[0])
    print(f"Single: {test_words[0]} -> {single_res}")

    print("\n=== 2. Testing Batch Transliteration ===")
    batch_res = transliterator.transliterate_batch(test_words)
    for orig, kana in batch_res.items():
        print(f"Batch : {orig} -> {kana}")
