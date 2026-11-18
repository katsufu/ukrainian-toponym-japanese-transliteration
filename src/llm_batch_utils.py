from collections import deque
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
    Stage B compact sub-batch recovery (chunked to <=10), Stage C single-item fallback
    at strict temperature=0.0, and sliding-window RPM pacing + accurate 429 classification.
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

    def __init__(self, provider_label: str, min_request_interval: float = 0.0, max_rpm: int = 0):
        self.provider_label = provider_label
        self.min_request_interval = float(min_request_interval)
        self.max_rpm = int(max_rpm)
        self._last_request_ts = 0.0
        self._request_window = deque()
        self._last_debug_info = ''

    def _pace_request(self):
        """Enforce both minimum inter-request spacing and a rolling 60-second RPM ceiling."""
        now = time.monotonic()
        wait_needed = 0.0

        if self.min_request_interval > 0 and self._last_request_ts > 0:
            since_last = now - self._last_request_ts
            if since_last < self.min_request_interval:
                wait_needed = max(wait_needed, self.min_request_interval - since_last)

        if self.max_rpm > 0:
            while self._request_window and (now - self._request_window[0]) >= 62.0:
                self._request_window.popleft()
            if len(self._request_window) >= self.max_rpm:
                window_wait = (self._request_window[0] + 62.0) - now
                if window_wait > wait_needed:
                    wait_needed = window_wait

        if wait_needed > 0:
            time.sleep(wait_needed)
            now = time.monotonic()

        self._last_request_ts = now
        if self.max_rpm > 0:
            while self._request_window and (now - self._request_window[0]) >= 62.0:
                self._request_window.popleft()
            self._request_window.append(now)

    @staticmethod
    def _extract_target_segment(text: str) -> str:
        """If the LLM outputs 'Ukrainian -> Katakana' or 'Ukrainian => Katakana', isolate the right-hand target side."""
        s = str(text or '').strip()
        if re.search(r'(?:->|=>|→|⇒)', s):
            s = re.split(r'(?:->|=>|→|⇒)', s)[-1].strip().strip('"\'')
        return s

    @classmethod
    def _clean_kana(cls, text: str) -> str:
        if not text:
            return ''
        s = unicodedata.normalize('NFKC', cls._extract_target_segment(text)).strip()
        # Normalize ASCII/fullwidth hyphens, dashes, or equals (with optional surrounding spaces) between Katakana into ＝
        s = re.sub(r'(?<=[ァ-ヶー])[\s　]*[=＝\-－―–—]+[\s　]*(?=[ァ-ヶ])', '＝', s)
        # Normalize spaces or repeated interpuncts between Katakana words into a single interpunct ・
        s = re.sub(r'(?<=[ァ-ヶー])[\s　・]+(?=[ァ-ヶ])', '・', s)
        # Retain only Katakana, prolonged sound mark ー, interpunct ・, and double hyphen ＝
        s = re.sub(r'[^ァ-ヶー・＝]', '', s)
        return s.strip('・＝')

    @classmethod
    def _has_cyrillic(cls, text: str) -> bool:
        target = cls._extract_target_segment(text)
        return bool(re.search(r'[А-Яа-яІіЇїЄєҐґ]', target))

    @staticmethod
    def _strip_accents(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn'
        ).strip()

    @classmethod
    def _is_valid_transliteration(cls, ukr_word: str, kana_word: str, raw_text: str = '') -> bool:
        if raw_text and cls._has_cyrillic(raw_text):
            return False
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

        # Salvage complete "key": "value" pairs (including malformed '"1": "ukr" => "kana"' lines) from truncated JSON outputs
        salvaged = {}
        arrow_line_pattern = re.compile(
            r'"(\d+)"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*"\s*(?::\s*"[^"\\]*(?:\\.[^"\\]*)*"\s*)?(?:=>|->|→|⇒)\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        )
        pair_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
        for k_raw, v_raw in pair_pattern.findall(text) + arrow_line_pattern.findall(text):
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
        exact/unaccented toponym keys, nested wrapper dicts, record objects, and lists.
        """
        flat_candidates = {}

        def _collect(obj):
            if isinstance(obj, dict):
                vals = list(obj.values())
                if len(obj) >= 2 and all(not str(k).isdigit() for k in obj.keys()) and all(isinstance(v, (str, int)) for v in vals):
                    str_vals = [str(v) for v in vals]
                    kana_vals = [cls._clean_kana(v) for v in str_vals if not cls._has_cyrillic(v) and cls._clean_kana(v)]
                    non_kana_vals = [v for v in str_vals if cls._has_cyrillic(v) or not cls._clean_kana(v)]
                    if len(kana_vals) == 1 and len(non_kana_vals) >= 1:
                        for nk in non_kana_vals:
                            flat_candidates[nk.strip()] = kana_vals[0]
                for k, v in obj.items():
                    if isinstance(v, str):
                        if cls._has_cyrillic(v):
                            continue
                        cleaned = cls._clean_kana(v)
                        if cleaned:
                            flat_candidates[str(k).strip()] = cleaned
                    elif isinstance(v, (dict, list)):
                        _collect(v)
            elif isinstance(obj, list):
                if len(obj) == len(toponyms_list) and all(isinstance(x, str) for x in obj):
                    for idx, x in enumerate(obj):
                        if cls._has_cyrillic(x):
                            continue
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
        Carefully distinguishes transient per-minute limits (e.g. Gemini 15 RPM containing boilerplate 'billing details')
        from genuine daily RPD / billing / zero-quota exhaustion.
        """
        msg = str(err)
        low = msg.lower()
        retry_sec = None
        m = re.search(r'retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s', low)
        if m:
            retry_sec = float(m.group(1))

        if any(tok in low for tok in ('invalid_api_key', 'api_key_invalid', 'incorrect api key', '401 ', '403 ', 'permissiondenied', 'authentication')):
            return 'fatal_auth', 0.0

        is_daily_or_zero_quota = any(
            tok in low for tok in (
                'insufficient_quota', 'perday', 'per_day', 'requestsperday',
                'generate_requests_per_day', 'limit: 0', 'billing not enabled', 'billing account'
            )
        )
        if is_daily_or_zero_quota:
            return 'quota_daily', 0.0

        if '429' in low or 'resource_exhausted' in low or 'rate limit' in low or 'too many requests' in low or 'perminute' in low:
            return 'rate_limit', retry_sec

        return 'transient', retry_sec

    def _handle_retry_exception(self, err: Exception, attempt: int, max_retries: int, context_label: str):
        """Handle API exceptions uniformly; return exception if rate-limited, else None. Raises QuotaExhaustedError on fatal/daily limits."""
        err_kind, retry_hint = self._classify_error(err)
        if err_kind in ('fatal_auth', 'quota_daily'):
            raise QuotaExhaustedError(f"{self.provider_label} API stopped ({err_kind}): {err}") from err
        if err_kind == 'rate_limit':
            wait_sec = (retry_hint + 3.0) if retry_hint is not None else min(65.0, 25.0 + attempt * 10.0)
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

    def _call_batch_api(self, prompt_text: str, force_json_mime: bool = True) -> str:
        raise NotImplementedError

    def _run_batch_request(self, items: list, max_retries: int = 4, prefer_plain_json: bool = False) -> dict:
        """
        Execute a batch API request at temperature=0.0:
        - Attempt 1 uses strict JSON MIME mode by default (or plain JSON when `prefer_plain_json=True` in sub-batch recovery).
        - If 200 OK returns 0 or <50% valid items, logs diagnostic sample and retries once
          with the alternate JSON MIME setting (still temperature=0.0) instead of repeating identical calls.
        - API exceptions / 429 rate limits still honor full `max_retries` backoff.
        """
        if not items:
            return {}
        indexed_input = {str(idx + 1): t for idx, t in enumerate(items)}
        input_json = json.dumps(indexed_input, ensure_ascii=False)
        prompt_text = self.BATCH_PROMPT.format(toponyms_json=input_json)

        best_results = {}
        last_rate_limit_err = None
        unparseable_count = 0

        for attempt in range(max_retries):
            use_json_mime = (unparseable_count == 1) if prefer_plain_json else (unparseable_count == 0)
            self._last_debug_info = ''
            try:
                self._pace_request()
                raw_text = self._call_batch_api(prompt_text, force_json_mime=use_json_mime)
                last_rate_limit_err = None
                parsed_json, _ = self._parse_json_or_salvage(raw_text)
                extracted = self._extract_batch_mapping(parsed_json, items)
                if len(extracted) >= len(best_results):
                    merged = dict(best_results)
                    merged.update(extracted)
                    best_results = merged
                else:
                    best_results.update(extracted)

                is_severely_incomplete = (
                    unparseable_count == 0
                    and len(items) >= 4
                    and len(best_results) < (len(items) // 2)
                )
                if len(best_results) > 0 and not is_severely_incomplete:
                    return best_results

                unparseable_count += 1
                preview = repr((raw_text or '')[:180])
                dbg = f" [{self._last_debug_info}]" if self._last_debug_info else ''
                print(
                    f"[{self.provider_label}] Empty/incomplete batch ({len(extracted)}/{len(items)} valid, "
                    f"json_mime={use_json_mime}, temp=0.0){dbg}. raw={preview}",
                    flush=True,
                )
                # Try at most 2 unparseable/severely-incomplete 200-OK attempts (strict JSON + plain JSON) before splitting smaller
                if unparseable_count >= 2:
                    break
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
        """Retrieve Japanese Katakana transliteration for a single toponym at temperature=0.0."""
        prompt_text = self.SINGLE_PROMPT.format(toponym=toponym)
        last_rate_limit_err = None
        invalid_ok_count = 0
        for attempt in range(max_retries):
            self._last_debug_info = ''
            try:
                self._pace_request()
                output = self._call_single_api(prompt_text)
                last_rate_limit_err = None
                kana = '' if self._has_cyrillic(output) else self._clean_kana(output)
                if self._is_valid_transliteration(toponym, kana, raw_text=output):
                    return kana
                invalid_ok_count += 1
                dbg = f" [{self._last_debug_info}]" if self._last_debug_info else ''
                print(
                    f"[{self.provider_label}] Invalid single output for '{toponym}' "
                    f"(attempt {attempt + 1}/{max_retries}, temp=0.0){dbg}: raw={repr((output or '')[:120])}",
                    flush=True,
                )
                # At temperature=0.0 with identical prompt, cap 200-OK invalid repeats at 2 to conserve RPD
                if invalid_ok_count >= 2:
                    break
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
        Multi-stage zero-drop batch retrieval without redundant duplicate retries:
        - Stage A: Primary 1-based JSON batch (with truncated JSON salvage & plain-JSON fallback)
        - Stage A2: Half-split (N/2 + N/2) if primary batch returned 0 items
        - Stage B: Compact sub-batch recovery split into chunks of <=10 (skips repeating failed 30-item chunks)
        - Stage C: Targeted single-item fallback at temperature=0.0
        """
        if not toponyms_list:
            return {}
        unique_items = list(dict.fromkeys(toponyms_list))
        if len(unique_items) == 1:
            t0 = unique_items[0]
            return {t0: self.transliterate(t0)}

        # Stage A: Primary batch request
        results = self._run_batch_request(unique_items, max_retries=max_retries)

        # Stage A2: If full batch returned 0 valid items, split once into interleaved even/odd half-batches
        # (separating adjacent accent-variant pairs like 'Краснопі́лля'/'Краснопілля' that trigger Cyrillic echoing)
        already_failed_size = len(unique_items) if not results else None
        if not results and len(unique_items) >= 10:
            even_half = unique_items[0::2]
            odd_half = unique_items[1::2]
            print(
                f"[{self.provider_label}] Primary batch returned 0 valid items; splitting into two sub-batches "
                f"({len(even_half)} + {len(odd_half)})...",
                flush=True,
            )
            half1 = self._run_batch_request(even_half, max_retries=max(3, max_retries - 1), prefer_plain_json=True)
            half2 = self._run_batch_request(odd_half, max_retries=max(3, max_retries - 1), prefer_plain_json=True)
            results.update(half1)
            results.update(half2)
            if not half1 or not half2:
                already_failed_size = max(len(even_half), len(odd_half))

        # Stage B: Sub-batch recovery for missing items using stride-interleaved compact chunks (max size 10)
        missing = [t for t in unique_items if not results.get(t)]
        if len(missing) > 1:
            sub_size = 10 if (already_failed_size and len(missing) >= already_failed_size) else min(15, len(missing))
            if sub_size >= 2:
                n_chunks = max(1, (len(missing) + sub_size - 1) // sub_size)
                # Ensure accent-variant pairs (e.g., 'Берегове'/'Берегове́') are always split into distinct chunks
                accent_counts = {}
                for t in missing:
                    k_norm = self._strip_accents(t)
                    accent_counts[k_norm] = accent_counts.get(k_norm, 0) + 1
                max_accent_mult = max(accent_counts.values())
                if max_accent_mult > n_chunks:
                    n_chunks = max_accent_mult
                eff_chunk_size = (len(missing) + n_chunks - 1) // n_chunks
                print(
                    f"[{self.provider_label}] Sub-batch recovery: requesting {len(missing)}/{len(unique_items)} "
                    f"missing items in {n_chunks} chunk(s) of <= {eff_chunk_size}...",
                    flush=True,
                )
                for c_idx in range(n_chunks):
                    sub_chunk = missing[c_idx::n_chunks]
                    if len(sub_chunk) == 1:
                        results[sub_chunk[0]] = self.transliterate(sub_chunk[0])
                    else:
                        recovered = self._run_batch_request(
                            sub_chunk, max_retries=max(3, max_retries - 1), prefer_plain_json=True
                        )
                        results.update(recovered)

        # Stage C: Final single-item fallback for stubborn items remaining after Stage B
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
