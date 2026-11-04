# Technical Architecture & Program Documentation

This document provides detailed technical specifications for the modules in `src/`, as well as guidelines for executing the pipeline on Uppsala University's **UPPMAX** HPC clusters.

---

## 🧩 Module Overview

```
src/
├── toponym_evaluate_pipeline.py          # Master orchestrator for end-to-end evaluation
├── ukrainian_g2p.py                      # Grapheme-to-Phoneme converter for Ukrainian
├── japanese_g2p.py                       # Katakana-to-IPA converter for Japanese
├── fed_evaluator.py                      # Articulatory Feature Edit Distance evaluator
├── perplexity_evaluator.py               # Phonotactic complexity evaluator (UniDic LM)
├── acoustic_distance_evaluator.py        # Meta MMS-1B Layer 7 + FastDTW evaluator
├── audio_generator.py                    # Dual-language TTS audio synthesizer
├── rulebased_transliteration.py          # MOFA guideline + established inflection detector
├── gpt_transliterator.py                 # OpenAI GPT-4o-mini zero-shot transliteration
├── gemini_transliteratior.py             # Google Gemini 3.5 Flash zero-shot transliteration
├── wikipedia_pageviews_inflection_point_detector.py # Kneedle algorithm & table exporter
└── toponym_visualizer.py                 # Multi-dimensional Altair interactive visualizer
```

---

## 🔬 Component Details

### 1. `ukrainian_g2p.py` (`UkrainianG2P`)
* **Underlying Model**: `UkrainianPhonemizer` from `ukrainian-tts-preprocessing` (Senyk et al., 2025).
* **Input**: Stressed Ukrainian Cyrillic string (e.g., `Ки́їв`, `Львів`).
* **Output**: Accurate Ukrainian IPA phoneme sequence (e.g., `kɪj'iw`, `lʲvʲiw`).
* **Key Functionality**: Resolves palatalization before front vowels, handles assimilation of voicing and place of articulation, and accounts for lexical stress shifts.

### 2. `japanese_g2p.py` (`JapaneseG2P`)
* **Underlying Model**: Epitran rule-based mapping (`jpn-Kana`).
* **Input**: Japanese Katakana string (e.g., `キーウ`, `リヴィウ`).
* **Output**: Japanese broad phonetic IPA representation (e.g., `kʲiːɯ`, `ɖʲibʲiɯ`).
* **Key Functionality**: Normalizes middle dots and whitespace, maps contracted sounds (拗音), geminate consonants (促音), and long vowels (長音).

### 3. `fed_evaluator.py` (`FEDEvaluator`)
* **Underlying Model**: PanPhon distinctive feature distance (Mortensen et al., 2016).
* **Methodology**: Rather than treating phonemes as atomic symbols (as in Levenshtein distance), PanPhon maps each IPA phoneme into a 22-dimensional articulatory feature vector ($\pm$voiced, $\pm$nasal, $\pm$continuant, etc.).
* **Formula**:
  $$\text{FED}(IPA_{ja}, IPA_{ukr}) = \frac{\sum \text{Feature Cost}}{\text{Path Length}}$$
* **Interpretation**: Lower scores indicate greater phonetic fidelity in terms of articulatory configurations.

### 4. `perplexity_evaluator.py` (`PerplexityEvaluator`)
* **Underlying Model**: Kneser-Ney Interpolated character 3-gram language model (`nltk.lm.KneserNeyInterpolated`).
* **Background Corpus**: Automatically extracts 50,000 pure Katakana vocabulary items (`[ァ-ヴー]{2,10}`) directly from the precompiled binary dictionary (`sys.dic`) of `unidic-lite` (Den et al., 2008). Zero external downloading required.
* **Interpretation**: Measures phonotactic complexity and pronounceability in Japanese. Non-canonical mora combinations (e.g., `ヴィ`, `ディ`) receive higher perplexity scores.

### 5. `acoustic_distance_evaluator.py` (`AcousticDistanceEvaluator`)
* **Underlying Model**: Meta MMS-1B (`facebook/mms-1b`, 1-billion parameter Wav2Vec2 encoder; Pratap et al., 2023).
* **Feature Extraction**: Extracts frame-level representations from **Layer 7** ($T \times 1280$), where speaker-specific acoustic noise is minimized and cross-lingual phonetic features are concentrated.
* **Alignment**: Compares variable-length Ukrainian and Japanese speech frames using **FastDTW** with cosine distance:
  $$\text{Dist}(A_{ukr}, A_{ja}) = \frac{1}{|P|} \sum_{(i, j) \in P} \left(1 - \frac{\vec{u}_i \cdot \vec{v}_j}{\|\vec{u}_i\|_2 \|\vec{v}_j\|_2}\right)$$
* **Batch Optimization**: Batches feature extraction with proper unpadded sequence slicing to avoid silence contamination in DTW.

### 6. `audio_generator.py` (`AudioGenerator`)
* **Ukrainian TTS**: ESPnet-based Ukrainian TTS (`ukrainian-tts`, Dmytro voice, model stress).
* **Japanese TTS**: PyOpenJTalk synthetic engine.
* **Output**: Cached 16kHz WAV audio files under `audio/ukrainian_ground_truth/` and `audio/japanese_candidates/`.

### 7. Transliterators (`gpt_transliterator.py` & `gemini_transliteratior.py`)
* **GPT**: OpenAI `gpt-4o-mini` with zero-shot Katakana prompt, `temperature=0.0`, structured JSON batching.
* **Gemini**: Google `gemini-3.5-flash-lite` with structured JSON batching, `temperature=0.0`.
* **Rate-Limit Handling**: Uses batch chunks of 30 toponyms to stay within API rate quotas.

### 8. `wikipedia_pageviews_inflection_point_detector.py` (`WikipediaViewsDetector`)
* **Algorithm**: Kneedle (Satopaa et al., 2011) over descending 2-year average monthly pageviews.
* **Output**:
  * Detects Knee Point threshold ($\approx 534$ views/month at Rank 500).
  * Exports `results/established_toponyms_table.csv` containing all 190 conventionalized toponyms.
  * Generates high-resolution visualization `results/wikipedia_pageview_knee_analysis.png`.

### 9. `toponym_visualizer.py` (`ToponymVisualizer`)
* **Altair / Vega-Lite Dashboard**:
  * Multi-method trade-off scatter plot (Acoustic Distance vs. Perplexity).
  * Grouped boxplots comparing FED, PPL, and Acoustic Distance distributions.
  * Interactive pageview rank curve with tooltips showing toponym names and monthly view counts.
  * Exports self-contained HTML: `results/evaluation_dashboard.html`.

---

## 💻 UPPMAX High-Performance Computing Execution

Uppsala University researchers should execute the pipeline on the **UPPMAX Rackham / Snowy** cluster.

### 1. Storage Quota Protection (Crucial)
Your home directory (`/home/username`) has a strict 32 GB quota. Do not install heavy models or conda environments in `/home/username`.

Always use your project allocation directory:
```bash
# Define your research project path
export PROJ_DIR="/proj/naiss202X-X-XX/username"
```

### 2. Environment Setup (One-Time)
Run on the login node or an interactive session:
```bash
# 1. Load system conda module
module load conda

# 2. Create conda environment on project disk
conda create -n ukr-toponym python=3.11 -y
source ~/.bashrc
conda activate ukr-toponym

# 3. Install packages without cluttering home pip cache
pip install --no-cache-dir -r requirements.txt
```

### 3. API Keys Configuration (In-Memory Security / Solution C)
To prevent API keys from ever being written to disk on shared HPC filesystems (where permissions or group shares could expose `~/.bashrc` or `.env`), enter the keys interactively in memory before launching the batch job. SLURM automatically inherits terminal environment variables into the batch context:
```bash
# Enter keys without terminal echo (no disk trace, no bash history)
read -sp "Enter OpenAI API Key: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
read -sp "Enter Gemini API Key: " GEMINI_API_KEY; echo; export GEMINI_API_KEY
```

### 4. Submitting the SLURM Job
```bash
# Submit batch job (inherits OPENAI_API_KEY and GEMINI_API_KEY from memory)
sbatch scripts/run_pipeline.sh

# Monitor status
squeue -u $USER
# or
jobinfo -u $USER

# Check output logs in real-time
tail -f logs/slurm_*.out
```

