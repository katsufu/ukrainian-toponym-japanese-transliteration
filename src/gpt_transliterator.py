import re
import json
import os
import unicodedata
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
            'Transliterate the following JSON object of Ukrainian toponyms (mapping index ID to Ukrainian toponym) into Japanese Katakana.\n'
            'Return ONLY a valid JSON object mapping each exact string index ID ("0", "1", ...) to its complete Japanese Katakana transliteration.\n\n'
            'Input JSON:\n{toponyms_json}')

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
                # Check if it's a record like {"id": "0", "katakana": "..."} or {"toponym": "...", "katakana": "..."}
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

        # Build lookup tables by index, exact toponym, and accent-stripped toponym
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

    # Retrieving transliteration from GPT API
    def transliterate(self, toponym, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    # parameter: temperature is for reproducibility
                    temperature=.0,
                    messages=[{'role': 'user', 'content': self.prompt.format(toponym=toponym)}]
                )
                output = response.choices[0].message.content.strip()
                kana = self._clean_kana(output)
                if self._is_valid_transliteration(toponym, kana):
                    return kana
            except Exception as e:
                print(f'GPT API Error for {toponym} (attempt {attempt + 1}/{max_retries}): {e}')
        return ''

    # Retrieving transliteration from GPT API in batch with per-item completeness verification
    def transliterate_batch(self, toponyms_list):
        if not toponyms_list:
            return {}
        try:
            indexed_input = {str(idx): t for idx, t in enumerate(toponyms_list)}
            input_json = json.dumps(indexed_input, ensure_ascii=False)
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=.0,
                response_format={"type": "json_object"},
                messages=[{'role': 'user', 'content': self.batch_prompt.format(toponyms_json=input_json)}]
            )
            raw_json = json.loads(response.choices[0].message.content.strip())
            results = self._extract_batch_mapping(raw_json, toponyms_list)
        except Exception as e:
            print(f'GPT Batch API Error: {e}')
            results = {}

        # Fill any missing or truncated items individually so batch output is 100% complete
        for t in toponyms_list:
            if t not in results or not results[t]:
                results[t] = self.transliterate(t)
        return results

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
