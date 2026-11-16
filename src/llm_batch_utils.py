import json
import re
import time
import unicodedata


class QuotaExhaustedError(RuntimeError):
    """Raised when an LLM API hits a non-transient quota, billing, or auth limit."""


class BaseLLMTransliterator:
    """
    Shared robust batch/single Ukrainian-to-Japanese Katakana transliteration engine.
    Implements 1-based JSON batch indexing ("1".."N"), truncated JSON salvage,
    Stage B missing-item sub-batch recovery, Stage C single-item fallback, and
    rate-limit / daily-quota classification.
    """

    SINGLE_PROMPT = (
        'You are an expert translator with Ukrainian and Japanese.\n'
        'Transliterate the following Ukrainian toponym into Japanese Katakana.\n'
        'Do output ONLY Japanese Katakana string \n\n'
        'Ukrainian Toponym: {toponym}'
    )

    BATCH_PROMPT = (
        'You are an expert translator with Ukrainian and Japanese.\n'
        'Transliterate the following JSON object of Ukrainian toponyms '
        '(mapping numeric string ID "1", "2", ... to Ukrainian toponym) into Japanese Katakana.\n'
        'Return ONLY a valid JSON object mapping exact ID string ("1", "2", ...) '
        'to its transliterated Japanese Katakana. Do not omit any ID.\n\n'
        'Input JSON:\n{toponyms_json}'
    )

    def __init__(self, provider_label: str, min_request_interval: float = 0.0):
        self.provider_label = provider_label
        self.min_request_interval = float(min_request_interval)
        self._last_request_ts = 0.0

    def _pace_request(self):
        """Enforce minimum interval between consecutive API calls to avoid RPM bursts."""
        if self.min_request_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if self._last_request_ts > 0 and elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_ts = time.monotonic()

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
        return ''.join(
            c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn'
        ).strip()

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

    @staticmethod
    def _parse_json_or_salvage(raw_text: str):
        """
        Parse LLM JSON response cleanly; if truncated near the end (JSONDecodeError),
        salvage all complete key-value string pairs already emitted before truncation.
        Returns (parsed_obj, is_complete_json).
        """
        text = str(raw_text or '').strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        candidates = [text]
        # Strip trailing commas before closing brace/bracket or isolate outermost braces
        candidates.append(re.sub(r',\s*([}\]])', r'\1', text))
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            sliced = text[first_brace:last_brace + 1]
            candidates.append(sliced)
            candidates.append(re.sub(r',\s*([}\]])', r'\1', sliced))

        for cand in candidates:
            try:
                return json.loads(cand), True
            except Exception:
                pass

        # Salvage complete "key": "value" pairs from truncated JSON outputs
        salvaged = {}
        pair_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
        for k_raw, v_raw in pair_pattern.findall(text):
            try:
                k_dec = json.loads(f'"{k_raw}"')
                v_dec = json.loads(f'"{v_raw}"')
            except Exception:
                k_dec, v_dec = k_raw, v_raw
            salvaged[str(k_dec).strip()] = v_dec
        return salvaged, False

    @classmethod
    def _extract_batch_mapping(cls, parsed_json, toponyms_list: list) -> dict:
        """
        Robustly extract {toponym: cleaned_kana} from LLM JSON output.
        Supports 1-based IDs ("1".."N"), defensive 0-based shift ("0".."N-1"),
        exact/unaccented toponym keys, nested wrapper dicts, and ordered lists.
        """
        flat_candidates = {}

        def _collect(obj):
            if isinstance(obj, dict):
                vals = list(obj.values())
                if len(obj) in (2, 3) and all(not str(k).isdigit() for k in obj.keys()) and all(isinstance(v, (str, int)) for v in vals):
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
                            flat_candidates[str(idx + 1)] = cleaned
                else:
                    for item in obj:
                        _collect(item)

        _collect(parsed_json)

        n_items = len(toponyms_list)
        # Detect accidental 0-based indexing ("0".."N-1") when "1".."N" was requested
        use_zero_based = ('0' in flat_candidates) and (str(n_items) not in flat_candidates)

        norm_lookup = {cls._strip_accents(k): v for k, v in flat_candidates.items()}
        results = {}
        for idx, t in enumerate(toponyms_list):
            t_str = str(t).strip()
            id_key = str(idx) if use_zero_based else str(idx + 1)
            kana = (
                flat_candidates.get(id_key)
                or flat_candidates.get(t_str)
                or norm_lookup.get(cls._strip_accents(t_str))
                or ''
            )
            if cls._is_valid_transliteration(t_str, kana):
                results[t] = kana
        return results

    @staticmethod
    def _classify_error(err: Exception) -> tuple:
        """
        Classify API exceptions into ('fatal_auth' | 'quota_daily' | 'rate_limit' | 'transient', retry_sec).
        """
        msg = str(err)
        low = msg.lower()
        retry_sec = None
        m = re.search(r'retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s', low)
        if m:
            retry_sec = float(m.group(1))

        if any(tok in low for tok in ('invalid_api_key', 'api_key_invalid', 'incorrect api key', '401 ', '403 ', 'permissiondenied', 'authentication')):
            return 'fatal_auth', 0.0

        if any(tok in low for tok in ('insufficient_quota', 'billing', 'perday', 'requestsperday', 'generate_requests_per_day', 'limit: 0')):
            return 'quota_daily', 0.0

        if '429' in low or 'resource_exhausted' in low or 'rate limit' in low or 'too many requests' in low:
            return 'rate_limit', retry_sec

        return 'transient', retry_sec

    def _handle_retry_exception(self, err: Exception, attempt: int, max_retries: int, context_label: str):
        """Handle API exceptions uniformly; return exception if rate-limited, else None. Raises QuotaExhaustedError on fatal/daily limits."""
        err_kind, retry_hint = self._classify_error(err)
        if err_kind in ('fatal_auth', 'quota_daily'):
            raise QuotaExhaustedError(f"{self.provider_label} API stopped ({err_kind}): {err}") from err
        if err_kind == 'rate_limit':
            wait_sec = (retry_hint + 2.0) if retry_hint is not None else min(65.0, 20.0 + attempt * 10.0)
            print(
                f"[{self.provider_label}] Rate Limit (429) on {context_label}. "
                f"Waiting {wait_sec:.1f}s before retry (attempt {attempt + 1}/{max_retries})...",
                flush=True,
            )
            time.sleep(wait_sec)
            return err
        print(
            f"[{self.provider_label}] API Error on {context_label} (attempt {attempt + 1}/{max_retries}): {err}",
            flush=True,
        )
        if attempt < max_retries - 1:
            time.sleep(min(15.0, 3.0 * (attempt + 1)))
        return None

    def _call_single_api(self, prompt_text: str) -> str:
        raise NotImplementedError

    def _call_batch_api(self, prompt_text: str) -> str:
        raise NotImplementedError

    def _run_batch_request(self, items: list, max_retries: int = 5) -> dict:
        """Execute a single batch API request for `items` with retry & salvaged parsing."""
        if not items:
            return {}
        indexed_input = {str(idx + 1): t for idx, t in enumerate(items)}
        input_json = json.dumps(indexed_input, ensure_ascii=False)
        prompt_text = self.BATCH_PROMPT.format(toponyms_json=input_json)

        best_results = {}
        last_rate_limit_err = None
        for attempt in range(max_retries):
            try:
                self._pace_request()
                raw_text = self._call_batch_api(prompt_text)
                parsed_json, _ = self._parse_json_or_salvage(raw_text)
                extracted = self._extract_batch_mapping(parsed_json, items)
                if len(extracted) > len(best_results):
                    best_results = extracted
                else:
                    best_results.update(extracted)
                if len(best_results) > 0:
                    return best_results
                print(
                    f"[{self.provider_label}] Empty/unparseable batch (0/{len(items)} valid). "
                    f"Retrying batch (attempt {attempt + 1}/{max_retries})...",
                    flush=True,
                )
            except Exception as e:
                rl_err = self._handle_retry_exception(e, attempt, max_retries, f"batch ({len(items)} items)")
                if rl_err is not None:
                    last_rate_limit_err = rl_err

        if not best_results and last_rate_limit_err is not None:
            raise QuotaExhaustedError(
                f"{self.provider_label} rate limit persisted across {max_retries} batch retries: {last_rate_limit_err}"
            )
        return best_results

    def transliterate(self, toponym, max_retries: int = 4) -> str:
        """Retrieve Japanese Katakana transliteration for a single toponym."""
        prompt_text = self.SINGLE_PROMPT.format(toponym=toponym)
        last_rate_limit_err = None
        for attempt in range(max_retries):
            try:
                self._pace_request()
                output = self._call_single_api(prompt_text)
                kana = self._clean_kana(output)
                if self._is_valid_transliteration(toponym, kana):
                    return kana
            except Exception as e:
                rl_err = self._handle_retry_exception(e, attempt, max_retries, f"'{toponym}'")
                if rl_err is not None:
                    last_rate_limit_err = rl_err
        if last_rate_limit_err is not None:
            raise QuotaExhaustedError(
                f"{self.provider_label} rate limit persisted on '{toponym}': {last_rate_limit_err}"
            )
        return ''

    def transliterate_batch(self, toponyms_list: list, max_retries: int = 5) -> dict:
        """
        Multi-stage zero-drop batch retrieval:
        - Stage A: Primary 1-based JSON batch (with truncated JSON salvage)
        - Stage B: Compact sub-batch recovery for any missing items (up to 2 passes)
        - Stage C: Targeted single-item fallback only for stubborn leftovers
        """
        if not toponyms_list:
            return {}
        unique_items = list(dict.fromkeys(toponyms_list))
        if len(unique_items) == 1:
            t0 = unique_items[0]
            return {t0: self.transliterate(t0)}

        # Stage A: Primary batch request
        results = self._run_batch_request(unique_items, max_retries=max_retries)

        # If a large batch returned 0 items due to formatting (not quota), split once into half-batches
        if not results and len(unique_items) >= 10:
            mid = len(unique_items) // 2
            print(
                f"[{self.provider_label}] Primary batch returned 0 valid items; retrying as two sub-batches "
                f"({mid} + {len(unique_items) - mid})...",
                flush=True,
            )
            results.update(self._run_batch_request(unique_items[:mid], max_retries=max(2, max_retries - 1)))
            results.update(self._run_batch_request(unique_items[mid:], max_retries=max(2, max_retries - 1)))

        # Stage B: Sub-batch recovery for missing items (batches all missing items together, max 2 passes)
        for recovery_pass in range(2):
            missing = [t for t in unique_items if not results.get(t)]
            if len(missing) <= 1:
                break
            print(
                f"[{self.provider_label}] Sub-batch recovery pass {recovery_pass + 1}: "
                f"requesting {len(missing)}/{len(unique_items)} missing items in 1 batch...",
                flush=True,
            )
            recovered = self._run_batch_request(missing, max_retries=max(2, max_retries - 1))
            if not recovered:
                break
            results.update(recovered)

        # Stage C: Final single-item fallback for stubborn items remaining after sub-batch recovery
        still_missing = [t for t in unique_items if not results.get(t)]
        if still_missing:
            if not results and len(still_missing) >= 10:
                probe = still_missing[0]
                probe_res = self.transliterate(probe)
                if not probe_res:
                    print(
                        f"[{self.provider_label}] WARNING: Batch and single-item probe both failed; "
                        f"skipping remaining {len(still_missing) - 1} items in this chunk.",
                        flush=True,
                    )
                    return {t: results.get(t, '') for t in toponyms_list}
                results[probe] = probe_res
                still_missing = still_missing[1:]
            if still_missing:
                print(
                    f"[{self.provider_label}] Recovering {len(still_missing)} stubborn missing item(s) individually...",
                    flush=True,
                )
                for t in still_missing:
                    results[t] = self.transliterate(t)
                    if not results[t]:
                        print(f"[{self.provider_label}] WARNING: Empty transliteration remained for '{t}'.", flush=True)

        return {t: results.get(t, '') for t in toponyms_list}

# Note: BaseLLMTransliterator is an abstract base class (_call_single_api / _call_batch_api are
# implemented by provider subclasses). Standalone __main__ smoke tests are provided in:
#   - src/gpt_transliterator.py      (Method B: OpenAI GPT API single & batch test)
#   - src/gemini_transliteratior.py  (Method C: Google Gemini API single & batch test)
