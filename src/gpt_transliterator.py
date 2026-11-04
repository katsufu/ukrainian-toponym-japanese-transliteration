import re
import json
import os
import openai

# Transliteration B: Zero-shot transliteration with OpenAI GPT API
class GPTTransliteration:

    def __init__(self, model_name='gpt-4o-mini'):
        api_key = os.environ.get('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.prompt = (
            'You are an expert translator with Ukrainian and Japanese.\n'
            'Transliterate the following Ukrainian toponym into Japanese Katakana.\n'
            'Do output ONLY Japanese string \n\n'
            'Ukrainian Toponym: {toponym}')
        self.batch_prompt = (
            'You are an expert translator with Ukrainian and Japanese.\n'
            'Transliterate the following JSON list of Ukrainian toponyms into Japanese Katakana.\n'
            'Return ONLY a valid JSON object mapping each original toponym to its Katakana transliteration.\n\n'
            'Input JSON:\n{toponyms_json}')

    # Retrieving transliteration from GPT API
    def transliterate(self, toponym):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                # parameter: temperature is for reproducibility
                temperature=.0,
                messages=[{'role': 'user', 'content': self.prompt.format(toponym=toponym)}]
            )
            output = response.choices[0].message.content.strip()
            # Remaining only Katakana, long vowel mark, and interpunct
            return re.sub(r'[^ァ-ヶー・]', '', output)
        except Exception as e:
            print(f'GPT API Error for {toponym}: {e}')
            return ''

    # Retrieving transliteration from GPT API
    def transliterate_batch(self, toponyms_list):
        try:
            input_json = json.dumps(toponyms_list, ensure_ascii=False)
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=.0,
                response_format={"type": "json_object"},
                messages=[{'role': 'user', 'content': self.batch_prompt.format(toponyms_json=input_json)}]
            )
            res_dict = json.loads(response.choices[0].message.content.strip())
            return {k: re.sub(r'[^ァ-ヶー・]', '', str(v)) for k, v in res_dict.items()}
        except Exception as e:
            print(f'GPT Batch API Error: {e}')
            return {t: self.transliterate(t) for t in toponyms_list}

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
