import re
import json
import time
import os
import google.generativeai as googleai

# Transliteration C: Zero-shot transliteration with Google Gemini API
class GeminiTransliteration:

    def __init__(self, model_name='gemini-3.5-flash-lite'):
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
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
    def transliterate(self, toponym, max_retries=3):
        prompt_text = self.prompt.format(toponym=toponym)
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt_text,
                    generation_config={'temperature': .0}
                )
                output = response.text.strip()
                # Rate limiting protection for free tier (<= 13.3 RPM)
                time.sleep(4.5)
                return re.sub(r'[^ァ-ヶー・]', '', output)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    wait_time = 32.0 + attempt * 5.0
                    m = re.search(r'retry in ([0-9]+(?:\.[0-9]+)?)s', err_str, re.IGNORECASE)
                    if m:
                        wait_time = float(m.group(1)) + 2.0
                    print(f"Gemini Rate Limit (429) hit for {toponym}. Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Gemini API Error for {toponym}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(3.0)
                    else:
                        return ""
        return ""

    # Retrieving transliteration from Gemini API with batch
    def transliterate_batch(self, toponyms_list, max_retries=5):
        input_json = json.dumps(toponyms_list, ensure_ascii=False)
        prompt_text = self.batch_prompt.format(toponyms_json=input_json)

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt_text,
                    generation_config={'temperature': .0, 'response_mime_type': 'application/json'}
                )
                text = response.text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                res_dict = json.loads(text)
                # Rate limiting protection for free tier (<= 13.3 RPM, comfortably below the 15 RPM limit)
                time.sleep(4.5)
                return {k: re.sub(r'[^ァ-ヶー・]', '', str(v)) for k, v in res_dict.items()}
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    wait_time = 32.0 + attempt * 5.0
                    m = re.search(r'retry in ([0-9]+(?:\.[0-9]+)?)s', err_str, re.IGNORECASE)
                    if m:
                        wait_time = float(m.group(1)) + 2.0
                    print(f"Gemini Batch Rate Limit (429) hit. Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f'Gemini Batch API Error: {e}')
                    if attempt < max_retries - 1:
                        time.sleep(3.0)
                    else:
                        break

        # Fallback to single requests with rate limiting if batch completely failed
        print("Gemini Batch failed after retries. Falling back to single requests with rate limiting...")
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
