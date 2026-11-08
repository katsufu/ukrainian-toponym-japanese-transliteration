import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# library for inflection point detection
from kneed import KneeLocator

# library for G2P and FED
import panphon.distance

# library for TTS
import pyopenjtalk
import scipy.io.wavfile
from ukrainian_tts.tts import TTS, Voices, Stress

# library for phonotactic metrics
import nltk
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

# library for Meta MMS & DTW
import torch
import torchaudio
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

# LLM API
import openai
import google.generativeai as googleai

# Import modules for the pipeline
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wikipedia_pageviews_inflection_point_detector import WikipediaViewsDetector
from ukrainian_g2p import UkrainianG2P
from japanese_g2p import JapaneseG2P
from rulebased_transliteration import RuleBasedTransliteration
from gpt_transliterator import GPTTransliteration
from gemini_transliteratior import GeminiTransliteration
from audio_generator import AudioGenerator
from fed_evaluator import FEDEvaluator
from perplexity_evaluator import PerplexityEvaluator
from acoustic_distance_evaluator import AcousticDistanceEvaluator

## Orchestrator for pipeline
class ToponymEvaluatePipeline:

    def __init__(self, corpus=None, pageviews_threshold=100.0, use_batch=True, batch_size=30):
        print(f'Initializing pipeline components (use_batch={use_batch}, batch_size={batch_size})...')

        self.use_batch = use_batch
        self.batch_size = batch_size

        # Instantiating G2P tools 
        self.ukrainian_g2p = UkrainianG2P()
        self.japanese_g2p = JapaneseG2P()

        # Instantiating transliterators
        self.rule_base_transliterator = RuleBasedTransliteration(pv_threshold=pageviews_threshold)
        self.gpt_transliterator = GPTTransliteration()
        self.gemini_transliterator = GeminiTransliteration()

        # Instantiating TTS
        self.audio_generator = AudioGenerator()

        # Instantiating evaluators
        self.fed = FEDEvaluator()
        self.ppl = PerplexityEvaluator(order=3)
        if corpus is not None and len(corpus) > 0:
            print(f"Training PerplexityEvaluator with custom corpus of {len(corpus)} entries...")
            self.ppl.fit(corpus)
        else:
            print("Training PerplexityEvaluator with UniDic-lite Katakana background corpus (50,000 words)...")
            self.ppl.fit_from_unidic(sample_size=50000)
            
        self.acoustic = AcousticDistanceEvaluator()

    def run_pipeline(self, df):
        print(f'Original total records: {len(df)}')

        # 1. For all entries, determine rule-base transliteration based on a Wikipedia pageviews threshold.
        print('Step 1: Applying Wikipedia pageviews threshold to rule-based transliteration...')
        translit_a_list = []
        for _, row in df.iterrows():
            translit_a_list.append(self.rule_base_transliterator.transliterate(row))
        df['transliteration_a_kana'] = translit_a_list

        # 2. Making unique combination between Japanese transliteration and Ukrainian toponyms
        unique_df = df[['toponym', 'toponym_with_accent', 'transliteration_a_kana']].drop_duplicates(
            subset=['toponym_with_accent', 'transliteration_a_kana']
        ).copy().reset_index(drop=True)
        # Assigning a unique ID for evaluation purposes
        unique_df['eval_id'] = [f'eval_{i+1}' for i in range(len(unique_df))]
        print(f'Step 2: Deduplicated down to {len(unique_df)} unique evaluation combinations!')
        
        # 3. Retrieving LLMs' transliteration (either with batch or sequential)
        #    Use deduplicated toponym_with_accent so LLMs receive stress marks (наголос)
        #    and batches never contain duplicate keys.
        accented_series = unique_df['toponym_with_accent'].fillna(unique_df['toponym']).astype(str).str.strip()
        toponym_list = list(dict.fromkeys(accented_series.tolist()))
        gpt_results = {}
        gemini_results = {}
        if self.use_batch:
            print(f'Step 3: Generating LLMs\' transliteration in BATCH mode (batch_size={self.batch_size}, unique_targets={len(toponym_list)})...')
            for i in tqdm(range(0, len(toponym_list), self.batch_size), desc="LLM Batches"):
                chunk = toponym_list[i:i + self.batch_size]
                # GPT Batch
                gpt_chunk = self.gpt_transliterator.transliterate_batch(chunk)
                gpt_results.update(gpt_chunk)
                # Gemini Batch
                gemini_chunk = self.gemini_transliterator.transliterate_batch(chunk)
                gemini_results.update(gemini_chunk)
        else:
            print(f'Step 3: Generating LLM transliterations in SEQUENTIAL mode (unique_targets={len(toponym_list)})...')
            for t in tqdm(toponym_list, desc="LLM Sequential"):
                gpt_results[t] = self.gpt_transliterator.transliterate(t)
                gemini_results[t] = self.gemini_transliterator.transliterate(t)
        unique_df['transliteration_b'] = [gpt_results.get(t, '') for t in accented_series]
        unique_df['transliteration_c'] = [gemini_results.get(t, '') for t in accented_series]
        
        # 4. G2P, TTS, evaluation
        print('Step 4: Running G2P, TTS, and Evaluations...')
        eval_records = []
        all_audio_paths = []
        for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="Evaluation"):
            e_id = row['eval_id']
            ukr_word = str(row['toponym_with_accent']) if pd.notna(row['toponym_with_accent']) else str(row['toponym'])
            kana_a = row['transliteration_a_kana']
            kana_b = row['transliteration_b']
            kana_c = row['transliteration_c']
            # G2P
            gt_ipa = self.ukrainian_g2p.convert(ukr_word)
            ipa_a = self.japanese_g2p.convert(kana_a)
            ipa_b = self.japanese_g2p.convert(kana_b) if kana_b else ''
            ipa_c = self.japanese_g2p.convert(kana_c) if kana_c else ''
            # TTS
            audio_gt = self.audio_generator.generate_ukrainian(ukr_word, e_id)
            audio_a = self.audio_generator.generate_japanese(kana_a, e_id, 'method_a')
            audio_b = self.audio_generator.generate_japanese(kana_b, e_id, 'method_b_gpt')
            audio_c = self.audio_generator.generate_japanese(kana_c, e_id, 'method_c_gemini')
            all_audio_paths.extend([audio_gt, audio_a, audio_b, audio_c])
            # FED
            fed_a = self.fed.evaluate(ipa_a, gt_ipa)
            fed_b = self.fed.evaluate(ipa_b, gt_ipa)
            fed_c = self.fed.evaluate(ipa_c, gt_ipa)
            # Perplexity
            ppl_a = self.ppl.evaluate(kana_a)
            ppl_b = self.ppl.evaluate(kana_b)
            ppl_c = self.ppl.evaluate(kana_c)
            eval_records.append({
                'toponym_with_accent': row['toponym_with_accent'],
                'transliteration_a_kana': kana_a,
                'ground_truth_ipa': gt_ipa,
                'transliteration_a_ipa': ipa_a,
                'transliteration_a_fed': fed_a,
                'transliteration_a_ppl': ppl_a,
                'audio_gt': audio_gt,
                'audio_a': audio_a,
                'transliteration_b': kana_b,
                'transliteration_b_ipa': ipa_b,
                'transliteration_b_fed': fed_b,
                'transliteration_b_ppl': ppl_b,
                'audio_b': audio_b,
                'transliteration_c': kana_c,
                'transliteration_c_ipa': ipa_c,
                'transliteration_c_fed': fed_c,
                'transliteration_c_ppl': ppl_c,
                'audio_c': audio_c,
            })
        eval_df = pd.DataFrame(eval_records)
        # MMS extracting feature representation & calculating acoustic distance
        print('Calculating Acoustic Distances...')
        if self.use_batch:
            feature_cache = self.acoustic.extract_features_batch(all_audio_paths, batch_size=16)
        else:
            feature_cache = None
        dist_a, dist_b, dist_c = [], [], []
        for _, r in eval_df.iterrows():
            dist_a.append(self.acoustic.evaluate(r['audio_gt'], r['audio_a'], feature_cache))
            dist_b.append(self.acoustic.evaluate(r['audio_gt'], r['audio_b'], feature_cache))
            dist_c.append(self.acoustic.evaluate(r['audio_gt'], r['audio_c'], feature_cache))
        eval_df['transliteration_a_acoustic'] = dist_a
        eval_df['transliteration_b_acoustic'] = dist_b
        eval_df['transliteration_c_acoustic'] = dist_c
        # Dropping unnecessary audio path columns
        eval_df = eval_df.drop(columns=['audio_gt', 'audio_a', 'audio_b', 'audio_c'])

        # 5. Merge the results back into the original dataframe to achieve a complete reconstruction
        print('Step 5: Merging evaluated unique records back into the full dataset...')
        final_df = pd.merge(
            df,
            eval_df,
            on=['toponym_with_accent', 'transliteration_a_kana'],
            how='left'
        )
        print(f'Successfully merged full dataset with {len(final_df)} rows and all evaluation metrics!')
        return final_df

if __name__ == '__main__':
    """
    Execution mode toggle flag
    True: Verify connectivity across all pipelines using only the first 5 items (a mock test that completes in a few seconds)
    False: Full-scale execution using all data
    """
    MOCK_MODE = False

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_candidates = [
        os.path.join(project_root, 'data', 'processed', 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'),
        os.path.join(project_root, 'data', 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'),
        'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'
    ]
    csv_file = next((p for p in csv_candidates if os.path.exists(p)), csv_candidates[0])
    if not os.path.exists(csv_file):
        results_dir = '.'

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # 1. Inflection point detection 
    optimizer = WikipediaViewsDetector()
    optimal_threshold = optimizer.fit_and_plot(
        df, 
        col_pv='avg_monthly_pv_2y', 
        output_image_path=os.path.join(results_dir, 'wikipedia_pageview_knee_analysis.png')
    )
    # Export established toponyms table for research paper
    optimizer.export_established_table(
        df, 
        knee_pv=optimal_threshold,
        output_csv=os.path.join(results_dir, 'established_toponyms_table.csv')
    )

    # 2. Initializing pipeline (using UniDic-lite Katakana background corpus)
    pipeline = ToponymEvaluatePipeline(
        corpus=None, 
        pageviews_threshold=optimal_threshold,
        use_batch=True,
        batch_size=5 if MOCK_MODE else 30
    )

    # 3. If conditions depending on MOCK_MODE boolean value
    if MOCK_MODE:
        print("\n[MOCK MODE] Testing pipeline with the first 5 records only...")
        eval_subset = df.head(5).copy()
        output_csv = os.path.join(results_dir, 'mock_evaluation_results.csv')
    else:
        print(f"\n[PRODUCTION MODE] Running full pipeline for all {len(df)} records...")
        eval_subset = df.copy()
        output_csv = os.path.join(results_dir, 'ukrainian_toponyms_evaluation_results.csv')

    # 4. Executing pipeline
    df_evaluated = pipeline.run_pipeline(eval_subset)

    # 5. Saving the result
    df_evaluated.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nCompleted! Results saved successfully to: {output_csv}")
