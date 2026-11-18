# Technical Architecture & Program Documentation

This document provides detailed technical specifications for the modules in `src/` that implement the multi-dimensional transliteration and evaluation pipeline.

---

## Module Overview

```
src/
├── toponym_evaluate_pipeline.py          # Master orchestrator for end-to-end evaluation
├── ukrainian_g2p.py                      # Ukrainian-to-IPA converter for Ukrainian
├── japanese_g2p.py                       # Deterministic mora-level Katakana-to-IPA converter
├── fed_evaluator.py                      # Articulatory Feature Edit Distance evaluator (PanPhon)
├── perplexity_evaluator.py               # Phonotactic complexity evaluator (UniDic Mora 3-gram LM)
├── acoustic_distance_evaluator.py        # Meta MMS (facebook/mms-1b) Layer 7 (CMVN) + FastDTW evaluator
├── audio_generator.py                    # Dual-language TTS audio synthesizer
├── rulebased_transliteration.py          # Hybrid rule-based + Wikipedia pageview threshold selector
├── llm_batch_utils.py                    # Shared 1-based batch JSON recovery, pacing & quota error handling
├── gpt_transliterator.py                 # OpenAI GPT-4o-mini zero-shot transliteration
├── gemini_transliteratior.py             # Google Gemini 3.5 Flash-Lite zero-shot transliteration
└── wikipedia_pageviews_inflection_point_detector.py # Kneedle algorithm & table exporter
```

---

## Component Details

### 1. `toponym_evaluate_pipeline.py` (`ToponymEvaluatePipeline`)
* **Role**: Master pipeline orchestrator coordinating threshold detection, deduplication, multi-run LLM generation (`--num-llm-runs 5`) with disk checkpointing, memoized G2P conversion, TTS synthesis, and 3-dimensional evaluation.
* **Workflow**:
  1. Applies the Wikipedia pageview knee-point threshold ($\ge 533.8$ views/month) across all $5,554$ records via `RuleBasedTransliteration`.
  2. Deduplicates into $1,853$ unique (`toponym_with_accent`, `transliteration_a_kana`) evaluation pairs (`eval_1` to `eval_1853`), spanning $1,821$ unique Ukrainian accented toponym strings.
  3. Generates LLM transliterations (Method B: GPT-4o-mini, Method C: Gemini 3.5 Flash-Lite) across $K=5$ independent runs in 1-based indexed JSON batches (`batch_size=60`), persisting each completed batch atomically to `results/llm_run_cache.json` (with `--clear-llm-cache`, `--clear-gemini-cache`, and `--clear-gpt-cache` flags for selective re-runs).
  4. Runs G2P, TTS audio synthesis, Feature Edit Distance (FED), Mora Perplexity (PPL), and batched Meta MMS (`facebook/mms-1b`) Layer 7 + FastDTW acoustic evaluation with identical-transliteration caching, computing the 5-run mean, standard deviation (`_std`), majority-vote transliteration, and self-consistency (`_agreement_rate`, `_exact_match`).
  5. Merges evaluated metrics back into the full $5,554$-row dataset and exports `results/ukrainian_toponyms_evaluation_results.csv`.

### 2. `ukrainian_g2p.py` (`UkrainianG2P`)
* **Underlying Model**: `Stressifier` from `ukrainian_word_stress` and `ipa-uk` phonological rules (Senyk et al., 2025).
* **Input**: Stressed Ukrainian Cyrillic string (e.g., `Ки́їв`, `Ві́нниця`, `Львів`).
* **Output**: Standardized post-1989 Ukrainian IPA phoneme sequence (e.g., `kɪːjiw`, `vʲiːnnɪt͡sʲɑ`, `lʲvʲiːw`).
* **Key Functionality**:
  * Preserves explicit combining acute stress marks (`\u0301`) and automatically assigns stress to monosyllabic words (excluding the unstressed preposition `за`) before invoking `Stressifier` only for unaccented words.
  * Normalizes deprecated pre-1989 Unicode affricate ligatures (`ʦ, ʣ, ʧ, ʤ`) into standard post-1989 IPA tie-bar sequences (`t͡s, d͡z, t͡ʃ, d͡ʒ`) recognized by PanPhon.
  * Converts stressed vowels (`'V`) into vowel length markers (`Vː`) to align with Japanese prolonged sound marks (`ー` $\rightarrow$ `ː`) without losing lexical stress information.

### 3. `japanese_g2p.py` (`JapaneseG2P`)
* **Underlying Model**: Deterministic mora-level phonological mapping (`DIGRAPH_MAP` with 66 digraphs and `MONOGRAPH_MAP`).
* **Input**: Japanese Katakana string (e.g., `キーウ`, `ヴィーンヌィツャ`, `リヴィウ`).
* **Output**: Phonetic IPA representation (e.g., `kʲiːɯ`, `vʲiːnnit͡sʲa`, `ɾʲivʲiɯ`).
* **Key Functionality**:
  * Accurately distinguishes palatalized glides (e.g., `リャ`=`ɾʲa`, `テャ`=`tʲa`, `デャ`=`dʲa`, `ツャ`=`t͡sʲa`), hard non-palatalized `/i/` digraphs (`ルィ`=`ɾi`, `ムィ`=`mi`, `ヌィ`=`ni`, `クィ`=`ki`, `グィ`=`ɡi`, `プィ`=`pi`, `ブィ`=`bi`, `トィ/ティ`=`ti`, `ドィ/ディ`=`di`, `スィ`=`si`, `ズィ`=`zi`, `ツィ`=`t͡si`), and labiodental `ヴ` (`v` / `vʲ`).
  * Resolves contextual gemination (`ッ` $\rightarrow$ doubles the following obstruent onset or inserts `t͡` before affricates), nasal place assimilation (`ン` $\rightarrow$ `m` before labials, `ŋ` before velars, `n` before alveolars, `ɴ` word-finally), and long vowels (`ー` $\rightarrow$ `ː`).

### 4. `fed_evaluator.py` (`FEDEvaluator`)
* **Underlying Model**: PanPhon articulatory distinctive feature distance (Mortensen et al., 2016).
* **Methodology**: Maps each IPA segment into a 24-dimensional articulatory feature vector ($\pm\text{syl}, \pm\text{son}, \pm\text{cons}, \pm\text{cont}, \pm\text{delrel}, \pm\text{lat}, \pm\text{nas}, \pm\text{voice}$, etc.) and computes dynamic programming edit distance weighted by subsegmental feature differences via `panphon.distance.Distance().feature_edit_distance`.
* **Interpretation**: Lower scores indicate greater phonetic fidelity in terms of articulatory configurations between the Japanese candidate IPA and Ukrainian Ground Truth IPA.

### 5. `perplexity_evaluator.py` (`PerplexityEvaluator`)
* **Underlying Model**: Mora-level Interpolated Modified Kneser-Ney (MKN) 3-gram language model implemented in pure Python with count-specific closed-form discounts ($D_1, D_2, D_{3+}$) and a 0th-order uniform prior over the complete 841-mora inventory (Chen & Goodman, 1999, Eq. 16–20 & Eq. 26).
* **Background Corpus**: Extracts 50,000 pure Katakana lexical items directly from the precompiled binary dictionary (`sys.dic`) of `unidic-lite` (Den et al., 2008), tokenized into phonotactic morae via `MORA_PATTERN = re.compile(r'[ァ-ヴ][ァィゥェォャュョヮ]?|ー')`.
* **Smoothing & OOV Resolution**: Uses true empirical N-gram frequencies at the highest order ($k=3$), unique continuation counts $N_{1+}(\bullet\, h\, w)$ at lower orders ($k \in \{1, 2\}$), and exact base interpolation with $P_0(w) = 1 / |V_{\text{all}}|$, guaranteeing finite perplexity scores while accurately reflecting Japanese phonotactic naturalness.

### 6. `acoustic_distance_evaluator.py` (`AcousticDistanceEvaluator`)
* **Underlying Model**: Meta MMS (`facebook/mms-1b`, `Wav2Vec2Model`; Pratap et al., 2024).
* **Signal & Feature Processing**:
  1. **Endpoint Silence Trimming (`_preprocess_waveform`)**: Resamples to $16\text{ kHz}$ mono, normalizes peak amplitude to $1.0$, and trims leading/trailing silence ($< 1\%$ peak amplitude with a $20\text{ ms}$ / $320$-sample margin, preserving at least $50\text{ ms}$ for the CNN feature extractor).
  2. **Attention-Masked Layer 7 Extraction**: Passes explicit `attention_mask` during batched GPU inference (`batch_size=16`) and slices exact unpadded frame lengths (`_get_feat_extract_output_lengths`) from **Layer 7** ($T \times 1280$).
  3. **Utterance-Level CMVN (`_cmvn`)**: Applies Cepstral / Feature Mean and Variance Normalization across the temporal dimension of each utterance to eliminate cross-synthesizer speaker timbre offsets between Ukrainian VITS (`Voices.Dmytro`) and Japanese Kokoro-82M (`jm_kumo`).
* **Alignment**: Computes length-normalized **FastDTW** cosine distance between the normalized Layer 7 frame trajectories:
  $$\text{Dist}(A_{ukr}, A_{ja}) = \frac{1}{|P|} \sum_{(i, j) \in P} \left(1 - \frac{\vec{u}_i \cdot \vec{v}_j}{\|\vec{u}_i\|_2 \|\vec{v}_j\|_2}\right)$$

### 7. `audio_generator.py` (`AudioGenerator`)
* **Ukrainian TTS**: ESPnet2 VITS Ukrainian synthesizer (`ukrainian-tts`, `Voices.Dmytro`, `Stress.Model`).
* **Japanese TTS**: Kokoro-82M Neural ONNX synthesizer (`kokoro-v1.0.fp16.onnx`, male voice `jm_kumo`), conditioned directly on phonetic IPA sequences produced by `JapaneseG2P` (`_g2p_to_kokoro_ipa`). Preserves both standard Katakana and Ukrainian-specific extended digraphs (`ルィ, ムィ, ヌィ, クィ, グィ, プィ, ブィ, トィ, ドィ, ツャ, イィ, テャ, デャ, ヴ`) with high-intelligibility neural speech resampled to 16 kHz.
* **Output**: Content-addressed, atomically written 16 kHz WAV audio files (`<eval_id>_<method>_<8char_md5>.wav`, validated for size $>44$ bytes) cached under `audio/ukrainian_ground_truth/` and `audio/japanese_candidates/`.

### 8. `rulebased_transliteration.py` (`RuleBasedTransliteration`)
* **Role**: Implements **Method A** (Hybrid Rule-Based + Sociolinguistic Threshold Selector).
* **Logic**: If a toponym has an existing Japanese Wikipedia article (`has_ja_wiki == True`) and its 2-year average monthly pageviews meet or exceed the Kneedle threshold (`avg_monthly_pv_2y >= pv_threshold`), returns the cleaned established Japanese Wikipedia title (`ja_wiki_title`); otherwise returns the accent-conditioned rule-based transliteration (`japanese_transliteration_by_accent`).

### 9. LLM Transliterators (`llm_batch_utils.py`, `gpt_transliterator.py` & `gemini_transliteratior.py`)
* **Models**:
  * **Method B (`GPTTransliteration`)**: OpenAI `gpt-4o-mini` (`temperature=0.0`, `response_format={"type": "json_object"}`, `timeout=60s`).
  * **Method C (`GeminiTransliteration`)**: Google `gemini-3.5-flash-lite` (`temperature=0.0`, `response_mime_type="application/json"`, `min_request_interval=5.5s`, `max_rpm=10`, `timeout=60s`).
* **Key Functionality (`BaseLLMTransliterator`)**:
  * Strictly maintains `temperature=0.0` and identical zero-shot prompts across all calls and retries, with 1-based numeric string keys (`"1", "2", ...`) in batched JSON prompts (`batch_size=60`), automatic 0-based shift correction (`"0".."N-1"`), and truncated JSON pair salvage.
  * Isolates right-hand Katakana targets when LLMs emit arrow-prefixed strings (`原語 -> カタカナ`, `_extract_target_segment`), rejects any Cyrillic residue (`_has_cyrillic`) or mid-word truncation (`_is_valid_transliteration`), and normalizes separators (`・` for spaces, `＝` for hyphens) via `_clean_kana`.
  * Implements multi-stage zero-drop recovery: Stage A (`json_mime=True` with single `json_mime=False` fallback) $\rightarrow$ Stage A2 (interleaved even/odd half-split) $\rightarrow$ Stage B (stride-interleaved compact sub-batches $\le 10$ separating accent-variant collisions) $\rightarrow$ Stage C (single-item fallback), plus sliding-window RPM pacing (`_pace_request`) and `QuotaExhaustedError` handling on daily RPD/auth exhaustion.

### 10. `wikipedia_pageviews_inflection_point_detector.py` (`WikipediaViewsDetector`)
* **Algorithm**: Kneedle knee-point detection (`KneeLocator(..., curve='convex', direction='decreasing', online=True)`; Satopaa et al., 2011) over descending 2-year average monthly Wikipedia pageviews (`avg_monthly_pv_2y`).
* **Output**:
  * Detects the Knee Point threshold ($533.8$ views/month at Rank $500$).
  * Exports `results/established_toponyms_table.csv` (190 unique established toponyms with Japanese Wikipedia articles above threshold) and `results/wikipedia_pageview_knee_analysis.png`.
