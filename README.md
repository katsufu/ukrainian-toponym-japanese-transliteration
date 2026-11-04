# Ukrainian Toponym Transliteration into Japanese: A Multi-Dimensional Quantitative Evaluation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official codebase and dataset for the Master's Research & Development project conducted at **Uppsala University (Department of Linguistics and Philology)**.

The project investigates the phonological, phonetic, and sociolinguistic trade-offs in transliterating Ukrainian toponyms into the Japanese Katakana writing system, evaluating rule-based transliterations against state-of-the-art Large Language Models (OpenAI GPT-4o-mini and Google Gemini 3.5 Flash).

---

## 📌 Research Overview

Transliterating Ukrainian geographical names into Japanese involves competing linguistic demands:
1. **Phonetic & Acoustic Fidelity**: Preserving native Ukrainian pronunciation (vowels, palatalized consonants, consonant clusters).
2. **Japanese Phonotactic Naturalness**: Conforming to Japanese morae constraints and ensuring pronounceability for native speakers.
3. **Sociolinguistic Entrenchment**: Accommodating conventionalized or historically established toponyms (e.g., *Kyiv* as `キーウ`, *Odesa* as `オデーサ`).

### Multi-Dimensional Evaluation Framework
To resolve the limitations of traditional string-matching metrics (e.g., character error rate / BLEU), our pipeline measures three complementary dimensions:

| Dimension | Metric | Implementation & Theoretical Rationale |
| :--- | :--- | :--- |
| **Phonetic Distance** | **Feature Edit Distance (FED)** | PanPhon articulatory distinctive features (Mortensen et al., 2016) comparing Ukrainian IPA to Japanese IPA. |
| **Phonotactic Naturalness** | **Perplexity (PPL)** | Kneser-Ney 3-gram language model trained on 50,000 Katakana lexical items extracted from UniDic-lite (Den et al., 2008). |
| **Acoustic Fidelity** | **Normalized Acoustic Distance** | Layer 7 representations from Meta MMS-1B (Massively Multilingual Speech; Pratap et al., 2023) aligned via Dynamic Time Warping (FastDTW, cosine distance). |
| **Sociolinguistic Factor** | **Knee Point Detection** | Inflection point analysis using Kneedle algorithm (Satopaa et al., 2011) over 2-year average Wikipedia pageviews to determine conventional vs. rule-based transliterations. |

---

## 📂 Repository Structure

```
ukrainian-toponym-transliteration/
├── src/                                      # Modular source code
│   ├── toponym_evaluate_pipeline.py          # Pipeline orchestrator
│   ├── ukrainian_g2p.py                      # Ukrainian G2P (Senyk et al., 2025)
│   ├── japanese_g2p.py                       # Japanese Katakana-to-IPA G2P (Epitran)
│   ├── fed_evaluator.py                      # Feature Edit Distance (PanPhon)
│   ├── perplexity_evaluator.py               # Phonotactic complexity (UniDic N-gram PPL)
│   ├── acoustic_distance_evaluator.py        # Meta MMS-1B + FastDTW acoustic distance
│   ├── audio_generator.py                    # Ukrainian (ESPnet) & Japanese (OpenJTalk) TTS
│   ├── rulebased_transliteration.py          # MOFA/Cabinet Secretariat rule-based system
│   ├── gpt_transliterator.py                 # OpenAI GPT-4o-mini (Single & Batch)
│   ├── gemini_transliteratior.py             # Google Gemini-3.5-flash-lite (Single & Batch)
│   ├── wikipedia_pageviews_inflection_point_detector.py # Kneedle knee-point detection
│   ├── toponym_visualizer.py                 # Altair interactive dashboard generator
│   └── README.md                             # Technical architecture & module documentation
│
├── data/                                     # Lexical and toponym datasets
│   ├── raw/                                  # Seed datasets
│   │   └── Ukrainian toponyms corpus.csv     # Raw Ukrainian toponyms & baseline Katakana
│   ├── intermediate/                         # Harversted Wikipedia/Wikidata metadata
│   │   └── ukrainian_toponyms_wikiinfo.csv   # Pageviews, oblast, coordinates, accents
│   └── processed/                            # Experiment benchmark datasets
│       ├── ukrainian_toponyms_wikiinfo_with_accent_ja.csv # Accent-expanded master dataset
│       └── established_toponyms_table.csv    # Toponyms exceeding the knee-point threshold
│
├── preprocessing/                            # Data harvesting & phonological expansion engines
│   ├── Ukrainian toponyms retrieval.py       # Wikipedia / Wikidata API harvesting script
│   ├── verify_and_expand_transliteration_en.py # Phonological rule & accent expansion script
│   └── README.md                             # Preprocessing pipeline documentation
│
├── results/                                  # Generated graphics and figures
│   ├── wikipedia_pageview_knee_analysis.png  # Knee-point detection plot (Rank: 500, 534 views/mo)
│   └── wikipedia_pageview_interactive.html   # Interactive Vega-Lite/Altair distribution plot
│
├── requirements.txt                          # Pinned direct dependencies
├── .gitignore                                # Excludes audio caches, models, and secrets
└── README.md                                 # This document
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies in a Python 3.11 environment:

```bash
git clone https://github.com/katsufu/ukrainian-toponym-japanese-transliteration.git
cd ukrainian-toponym-japanese-transliteration

# Create and activate conda environment
conda create -n ukr-toponym python=3.11 -y
conda activate ukr-toponym

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (LLM API Keys)
Configure your API keys in your environment (do not hardcode them in scripts):

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Running the Pipeline
You can test the connectivity of all components in **Mock Mode** (first 5 records, completes in seconds):

```bash
# Run pipeline (toggle MOCK_MODE = True/False in src/toponym_evaluate_pipeline.py)
python src/toponym_evaluate_pipeline.py
```

### 4. Interactive Visualizations
Generate and view the multi-dimensional evaluation dashboard in your browser:

```bash
python src/toponym_visualizer.py
```
Open `results/wikipedia_pageview_interactive.html` or `results/evaluation_dashboard.html` to explore toponyms interactively.

---

## 📖 References

* **Senyk, O., et al. (2025).** *Grapheme-to-Phoneme conversion for Ukrainian using linguistic stress prediction.*
* **Mortensen, D. R., et al. (2016).** *PanPhon: A Resource for Mapping IPA to Articulatory Feature Vectors.* COLING.
* **Den, Y., et al. (2008).** *The design and implementation of UniDic: A lexicon for morphological analysis.*
* **Pratap, V., et al. (2023).** *Scaling Speech Technology to 1,000+ Languages (Meta MMS).* arXiv:2305.13516.
* **Satopaa, V., et al. (2011).** *Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior.* IEEE.

