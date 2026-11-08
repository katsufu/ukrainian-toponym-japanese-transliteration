import re
import json
import time
import os
import unicodedata
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
            'Transliterate the following JSON object of Ukrainian toponyms (mapping index ID to Ukrainian toponym) into Japanese Katakana.\n'
            'Return ONLY a valid JSON object mapping each exact string index ID ("0", "1", ...) to its complete Japanese Katakana transliteration.\n\n'
            'Input JSON:\n{toponyms_json}'
        )

    @staticmethod
    def _clean_kana(text: str) -> str:
        if not text:
            return ''
        s = str(text).strip()
        # Normalize ASCII/fullwidth hyphens or equals between Katakana into ＝
        s = re.sub(r'(?<=[ァ-ヶー])[=＝\-－―]+(?=[ァ-ヶ])', '＝', s)
        # Normalize spaces between Katakana words into interpunct ・
        s = re.sub(r'(?<=[ァ-ヶー])[\s　]+(?=[ァ-ヶ])', '・', s)
        # Retain only Katakana, prolonged sound mark ー, interpunct ・, and double hyphen ＝
        s = re.sub(r'[^ァ-ヶー・＝]', '', s)
        return s.strip('・＝')

    @staticmethod
    def _strip_accents(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn').strip()

    @classmethod
    def _is_valid_transliteration(cls, ukr_word: str, kana_word: str) -> bool:
        if not kana_word:
            return False
        cyr_len = len(re.findall(r'[А-Яа-яІіЇїЄєҐґ]', str(ukr_word)))
        kana_len = len(re.findall(r'[ァ-ヶー]', str(kana_word)))
        if kana_len == 0:
            return False
        # Prevent mid-word JSON truncation (e.g., >=6 Cyrillic letters truncated to <=2 Katakana chars)
        if cyr_len >= 6 and kana_len < 3:
            return False
        return True

    @classmethod
    def _extract_batch_mapping(cls, parsed_json, toponyms_list: list) -> dict:
        """
        Robustly extract {toponym: cleaned_kana} from LLM JSON output whether keyed by
        index ("0", "1"), exact toponym, unaccented toponym, nested dict, or list of objects.
        """
        flat_candidates = {}

        def _collect(obj):
            if isinstance(obj, dict):
                vals = list(obj.values())
                if len(obj) in (2, 3) and all(isinstance(v, (str, int)) for v in vals):
                    str_vals = [str(v) for v in vals]
                    kana_vals = [cls._clean_kana(v) for v in str_vals if cls._clean_kana(v)]
                    non_kana_vals = [v for v in str_vals if not cls._clean_kana(v)]
                    if len(kana_vals) == 1 and len(non_kana_vals) >= 1:
                        for nk in non_kana_vals:
                            flat_candidates[nk.strip()] = kana_vals[0]
                for k, v in obj.items():
                    if isinstance(v, str):
                        cleaned = cls._clean_kana(v)
                        if cleaned:
                            flat_candidates[str(k).strip()] = cleaned
                    elif isinstance(v, (dict, list)):
                        _collect(v)
            elif isinstance(obj, list):
                if len(obj) == len(toponyms_list) and all(isinstance(x, str) for x in obj):
                    for idx, x in enumerate(obj):
                        cleaned = cls._clean_kana(x)
                        if cleaned:
                            flat_candidates[str(idx)] = cleaned
                else:
                    for item in obj:
                        _collect(item)

        _collect(parsed_json)

        norm_lookup = {cls._strip_accents(k): v for k, v in flat_candidates.items()}
        results = {}
        for idx, t in enumerate(toponyms_list):
            t_str = str(t).strip()
            kana = (
                flat_candidates.get(str(idx))
                or flat_candidates.get(t_str)
                or norm_lookup.get(cls._strip_accents(t_str))
                or ''
            )
            if cls._is_valid_transliteration(t_str, kana):
                results[t] = kana
        return results

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
                kana = self._clean_kana(output)
                if self._is_valid_transliteration(toponym, kana):
                    return kana
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

    # Retrieving transliteration from Gemini API with batch and completeness verification
    def transliterate_batch(self, toponyms_list, max_retries=5):
        if not toponyms_list:
            return {}
        indexed_input = {str(idx): t for idx, t in enumerate(toponyms_list)}
        input_json = json.dumps(indexed_input, ensure_ascii=False)
        prompt_text = self.batch_prompt.format(toponyms_json=input_json)

        results = {}
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
                raw_json = json.loads(text)
                # Rate limiting protection for free tier (<= 13.3 RPM, comfortably below the 15 RPM limit)
                time.sleep(4.5)
                extracted = self._extract_batch_mapping(raw_json, toponyms_list)
                results.update(extracted)
                # If all items in the batch were validly extracted, return immediately
                if len(results) == len(toponyms_list):
                    return results
                # If >70% succeeded in the batch, break and fill the few missing ones individually
                if len(results) >= max(1, int(0.7 * len(toponyms_list))):
                    break
                print(f"Gemini Batch incomplete ({len(results)}/{len(toponyms_list)} valid). Retrying batch (attempt {attempt + 1}/{max_retries})...")
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

        # Fill any remaining missing or truncated items individually with rate limiting
        missing = [t for t in toponyms_list if t not in results or not results[t]]
        if missing:
            print(f"Fetching {len(missing)} remaining items individually from Gemini with rate limiting...")
            for t in missing:
                results[t] = self.transliterate(t)
        return results

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
