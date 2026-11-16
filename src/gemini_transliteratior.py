import os
import sys
import google.generativeai as googleai

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from llm_batch_utils import BaseLLMTransliterator  # noqa: E402


# Transliteration C: Zero-shot transliteration with Google Gemini API
class GeminiTransliteration(BaseLLMTransliterator):

    def __init__(
        self,
        model_name='gemini-3.5-flash-lite',
        min_request_interval: float = 6.5,
        request_timeout: float = 60.0,
    ):
        super().__init__(provider_label='Gemini', min_request_interval=min_request_interval)
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            googleai.configure(api_key=api_key)
        self.model = googleai.GenerativeModel(model_name)
        self.request_timeout = request_timeout
        self.prompt = self.SINGLE_PROMPT
        self.batch_prompt = self.BATCH_PROMPT

    def _call_single_api(self, prompt_text: str) -> str:
        response = self.model.generate_content(
            prompt_text,
            generation_config={'temperature': .0},
            request_options={'timeout': self.request_timeout},
        )
        return (response.text or '').strip()

    def _call_batch_api(self, prompt_text: str) -> str:
        response = self.model.generate_content(
            prompt_text,
            generation_config={'temperature': .0, 'response_mime_type': 'application/json'},
            request_options={'timeout': self.request_timeout},
        )
        return (response.text or '').strip()


if __name__ == '__main__':
    transliterator = GeminiTransliteration()
    test_words = ["Ки́їв", "Львів", "Ви́года"]

    print("=== 1. Testing Single Transliteration ===")
    single_res = transliterator.transliterate(test_words[0])
    print(f"Single: {test_words[0]} -> {single_res}")

    print("\n=== 2. Testing Batch Transliteration ===")
    batch_res = transliterator.transliterate_batch(test_words)
    for orig, kana in batch_res.items():
        print(f"Batch : {orig} -> {kana}")
