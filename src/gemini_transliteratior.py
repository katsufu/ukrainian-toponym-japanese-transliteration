import re
import json
import time
import os
import google.generativeai as googleai

# Transliteration C: Zero-shot transliteration with Google Gemini API
class GeminiTransliteration:

    def __init__(self, model_name='gemini-3.5-flash-lite'):
        api_key = os.environ.get('GEMINI_API_KEY')
        googleai.configure(api_key=api_key)
        self.model = googleai.GenerativeModel(model_name)
        self.prompt = (
            'You are an expert translator with Ukrainian and Japanese.\n'
            'Transliterate the following Ukrainian toponym into Japanese Katakana.\n'
            'Do output ONLY Japanese string \n\n'
            'Ukrainian Toponym: {toponym}'
        )
        self.batch_prompt = (
            'You are an expert translator with Ukrainian and Japanese.\n'
            'Transliterate the following JSON list of Ukrainian toponyms into Japanese Katakana.\n'
            'Return ONLY a valid JSON object mapping each original toponym to its Katakana transliteration.\n\n'
            'Input JSON:\n{toponyms_json}'
        )

    # Retrieving transliteration from Gemini API
    def transliterate(self, toponym):
        try:
            response = self.model.generate_content(
                self.prompt.format(toponym=toponym),
                generation_config={'temperature': .0}
            )
            output = response.text.strip()
            # Rate limiting protection
            time.sleep(.05)
            return re.sub(r'[^ァ-ヶー・]', '', output)
        except Exception as e:
            print(f"Gemini API Error for {toponym}: {e}")
            return ""

    # Retrieving transliteration from Gemini API with batch
    def transliterate_batch(self, toponyms_list):
        try:
            input_json = json.dumps(toponyms_list, ensure_ascii=False)
            response = self.model.generate_content(
                self.batch_prompt.format(toponyms_json=input_json),
                generation_config={'temperature': .0, 'response_mime_type': 'application/json'}
            )
            res_dict = json.loads(response.text.strip())
            time.sleep(0.1) 
            return {k: re.sub(r'[^ァ-ヶー・]', '', str(v)) for k, v in res_dict.items()}
        except Exception as e:
            print(f'Gemini Batch API Error: {e}')
            return {t: self.transliterate(t) for t in toponyms_list}

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