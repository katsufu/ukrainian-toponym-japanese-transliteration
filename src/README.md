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
├── acoustic_distance_evaluator.py        # WavLM-Base-Plus Layer 7 (CMVN) + FastDTW evaluator
├── audio_generator.py                    # Dual-language TTS audio synthesizer
├── rulebased_transliteration.py          # Hybrid rule-based + Wikipedia pageview threshold selector
├── gpt_transliterator.py                 # OpenAI GPT-4o-mini zero-shot transliteration
├── gemini_transliteratior.py             # Google Gemini 3.5 Flash zero-shot transliteration
└── wikipedia_pageviews_inflection_point_detector.py # Kneedle algorithm & table exporter
```

---

## Component Details

### 1. `toponym_evaluate_pipeline.py` (`ToponymEvaluatePipeline`)
* **Role**: Master pipeline orchestrator coordinating threshold detection, deduplication, LLM generation, G2P conversion, TTS synthesis, and 3-dimensional evaluation.
* **Workflow**:
  1. Applies the Wikipedia pageview knee-point threshold ($\ge 533.8$ views/month) across all $5,554$ records via `RuleBasedTransliteration`.
  2. Deduplicates into $1,821$ unique (`toponym_with_accent`, `transliteration_a_kana`) evaluation pairs (`eval_1` to `eval_1821`).
  3. Generates LLM transliterations (Method B: GPT-4o-mini, Method C: Gemini-3.5-flash-lite) in indexed JSON batches (`batch_size=30`).
  4. Runs G2P, TTS audio synthesis, Feature Edit Distance (FED), Mora Perplexity (PPL), and batched WavLM Layer 7 + FastDTW acoustic evaluation.
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
* **Underlying Model**: Mora-level Kneser-Ney interpolated 3-gram language model (`nltk.lm.KneserNeyInterpolated`, `discount=0.75`) with uniform unigram inventory prior interpolation (Chen & Goodman, 1999).
* **Background Corpus**: Extracts 50,000 pure Katakana lexical items directly from the precompiled binary dictionary (`sys.dic`) of `unidic-lite` (Den et al., 2008), tokenized into phonotactic morae via `_MORA_RE = re.compile(r'[ァ-ヴ][ァィゥェォャュョヮ]?|ー')`.
* **Smoothing & OOV Resolution**: Appends a single uniform pseudo-observation across the complete 841-mora inventory (`_build_full_mora_inventory()`), guaranteeing finite, non-degenerate perplexity scores (`inf` = 0) while penalizing non-canonical foreign mora combinations (e.g., `ルィ`, `テャ`, `ヴャ`) in proportion to their rarity in Japanese phonotactics.

### 6. `acoustic_distance_evaluator.py` (`AcousticDistanceEvaluator`)
* **Underlying Model**: Microsoft WavLM-Base-Plus (`microsoft/wavlm-base-plus`; Chen et al., 2022).
* **Signal & Feature Processing**:
  1. **Endpoint Silence Trimming (`_preprocess_waveform`)**: Trims leading and trailing silence ($-35\text{ dB}$ threshold with a $25\text{ ms}$ collar) and applies peak normalization prior to SSL encoding.
  2. **Attention-Masked Layer 7 Extraction**: Passes explicit `attention_mask` during batched GPU inference (`batch_size=16`) and slices exact unpadded frame lengths (`_compute_feat_len`) from **Layer 7** ($T \times 768$).
  3. **Utterance-Level CMVN (`_cmvn`)**: Applies Cepstral Mean and Variance Normalization across the temporal dimension of each utterance to eliminate cross-gender/cross-engine speaker timbre bias between Ukrainian female TTS and Japanese male TTS.
* **Alignment**: Computes length-normalized **FastDTW** cosine distance between the normalized Layer 7 frame trajectories:
  $$\text{Dist}(A_{ukr}, A_{ja}) = \frac{1}{|P|} \sum_{(i, j) \in P} \left(1 - \frac{\vec{u}_i \cdot \vec{v}_j}{\|\vec{u}_i\|_2 \|\vec{v}_j\|_2}\right)$$

### 7. `audio_generator.py` (`AudioGenerator`)
* **Ukrainian TTS**: ESPnet VITS Ukrainian synthesizer (`ukrainian-tts`, `Voices.Tetiana`, `Stress.Dictionary`), converting combining acute stress marks (`\u0301`) to `+` stress markers prior to synthesis.
* **Japanese TTS**: `pyopenjtalk` HTS engine with full-context label manipulation (`_apply_hts_mods`). Substitutes proxy morae before MeCab parsing and rewrites the resulting 5-phoneme HTS contexts (`p1^p2-p3+p4=p5`) prior to waveform synthesis so that extended Ukrainian-specific Katakana digraphs (`ルィ, ムィ, ヌィ, クィ, グィ, プィ, ブィ, トィ, ドィ, ツャ, イィ, テャ, デャ`) are synthesized as true 1-mora syllables rather than splitting into 2 morae.
* **Output**: Content-addressed 16kHz WAV audio files (`<eval_id>_<8char_md5>.wav`) cached under `audio/ukrainian_ground_truth/` and `audio/japanese_candidates/`.

### 8. `rulebased_transliteration.py` (`RuleBasedTransliteration`)
* **Role**: Implements **Method A** (Hybrid Rule-Based + Sociolinguistic Threshold Selector).
* **Logic**: If a toponym has an existing Japanese Wikipedia article (`has_ja_wiki == True`) and its 2-year average monthly pageviews meet or exceed the Kneedle threshold (`avg_monthly_pv_2y >= pv_threshold`), returns the cleaned established Japanese Wikipedia title (`ja_wiki_title`); otherwise returns the accent-conditioned rule-based transliteration (`japanese_transliteration_by_accent`).

### 9. LLM Transliterators (`gpt_transliterator.py` & `gemini_transliteratior.py`)
* **Models**:
  * **Method B (`GPTTransliteration`)**: OpenAI `gpt-4o-mini` (`temperature=0.0`, `response_format={"type": "json_object"}`).
  * **Method C (`GeminiTransliteration`)**: Google `gemini-3.5-flash-lite` (`temperature=0.0`, `response_mime_type="application/json"`).
* **Key Functionality**:
  * Uses integer string keys (`"0", "1", ...`) in batched JSON prompts (`batch_size=30`) to prevent key mismatch caused by Unicode stress mark dropping.
  * Normalizes word separators (`・` for spaces, `＝` for hyphens) via `_clean_kana` and validates output length against Cyrillic character counts via `_is_valid_transliteration` to reject mid-word JSON truncations, automatically retrying any missing or truncated items individually.

### 10. `wikipedia_pageviews_inflection_point_detector.py` (`WikipediaViewsDetector`)
* **Algorithm**: Kneedle knee-point detection (Satopaa et al., 2011) over log-transformed descending 2-year average monthly Wikipedia pageviews (`avg_monthly_pv_2y`).
* **Output**:
  * Detects the Knee Point threshold ($533.8$ views/month at Rank $213$).
  * Exports `results/established_toponyms_table.csv` (190 unique established toponyms across 213 records) and `results/wikipedia_pageview_knee_analysis.png`.
