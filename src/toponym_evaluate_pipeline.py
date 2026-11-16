import argparse
from collections import Counter
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from wikipedia_pageviews_inflection_point_detector import WikipediaViewsDetector
from ukrainian_g2p import UkrainianG2P
from japanese_g2p import JapaneseG2P
from rulebased_transliteration import RuleBasedTransliteration
from llm_batch_utils import QuotaExhaustedError
from gpt_transliterator import GPTTransliteration
from gemini_transliteratior import GeminiTransliteration
from audio_generator import AudioGenerator
from fed_evaluator import FEDEvaluator
from perplexity_evaluator import PerplexityEvaluator
from acoustic_distance_evaluator import AcousticDistanceEvaluator


## Orchestrator for pipeline
class ToponymEvaluatePipeline:

    def __init__(
        self,
        corpus=None,
        pageviews_threshold: float = 100.0,
        use_batch: bool = True,
        batch_size: int = 60,
        num_llm_runs: int = 5,
        base_audio_dir: str = 'audio',
        llm_cache_path: str = None,
    ):
        print(
            f'Initializing pipeline components (use_batch={use_batch}, '
            f'batch_size={batch_size}, num_llm_runs={num_llm_runs})...',
            flush=True,
        )

        self.use_batch = use_batch
        self.batch_size = batch_size
        self.num_llm_runs = max(int(num_llm_runs), 1)
        self.llm_cache_path = llm_cache_path

        # Instantiating G2P tools
        self.ukrainian_g2p = UkrainianG2P()
        self.japanese_g2p = JapaneseG2P()

        # Instantiating transliterators
        self.rule_base_transliterator = RuleBasedTransliteration(pv_threshold=pageviews_threshold)
        self.gpt_transliterator = GPTTransliteration()
        self.gemini_transliterator = GeminiTransliteration()

        # Instantiating TTS (Ukrainian VITS + Japanese Kokoro-82M Neural ONNX jm_kumo)
        self.audio_generator = AudioGenerator(base_audio_dir=base_audio_dir, japanese_voice='jm_kumo')

        # Instantiating evaluators
        self.fed = FEDEvaluator()
        self.ppl = PerplexityEvaluator(order=3)
        if corpus is not None and len(corpus) > 0:
            print(f"Training PerplexityEvaluator (Interpolated MKN) with custom corpus of {len(corpus)} entries...")
            self.ppl.fit(corpus)
        else:
            print("Training PerplexityEvaluator (Interpolated MKN) with UniDic-lite Katakana corpus (50,000 words)...")
            self.ppl.fit_from_unidic(sample_size=50000)

        self.acoustic = AcousticDistanceEvaluator()

    @staticmethod
    def _mean_and_std(values: list) -> tuple:
        valid = [
            float(v) for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        if not valid:
            return float('nan'), float('nan')
        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid, ddof=1)) if len(valid) >= 2 else 0.0
        return mean_val, std_val

    @classmethod
    def _aggregate_llm_runs(
        cls,
        kana_runs: list,
        ipa_runs: list,
        fed_runs: list,
        ppl_runs: list,
        acoustic_runs: list,
        num_runs: int,
    ) -> dict:
        """
        Aggregate K independent LLM runs for a single toponym:
        - Majority-vote (mode) Katakana string and corresponding IPA (tie-broken by first appearance)
        - Self-consistency metrics: num_unique, agreement_rate, exact_match across K runs
        - 5-run mean and sample standard deviation (SD) for FED, PPL, and Acoustic Distance
        """
        valid_pairs = [(k, ipa) for k, ipa in zip(kana_runs, ipa_runs) if k and str(k).strip()]
        if valid_pairs:
            counts = Counter(k for k, _ in valid_pairs)
            mode_kana, mode_count = counts.most_common(1)[0]
            mode_ipa = next(ipa for k, ipa in valid_pairs if k == mode_kana)
            num_unique = len(counts)
            agreement_rate = float(mode_count) / len(valid_pairs)
            exact_match = (len(valid_pairs) == num_runs) and (num_unique == 1)
        else:
            mode_kana = ''
            mode_ipa = ''
            num_unique = 0
            agreement_rate = float('nan')
            exact_match = False

        fed_mean, fed_std = cls._mean_and_std(fed_runs)
        ppl_mean, ppl_std = cls._mean_and_std(ppl_runs)
        acoustic_mean, acoustic_std = cls._mean_and_std(acoustic_runs)

        return {
            'kana': mode_kana,
            'ipa': mode_ipa,
            'fed_mean': fed_mean,
            'fed_std': fed_std,
            'ppl_mean': ppl_mean,
            'ppl_std': ppl_std,
            'acoustic_mean': acoustic_mean,
            'acoustic_std': acoustic_std,
            'num_unique': num_unique,
            'agreement_rate': agreement_rate,
            'exact_match': exact_match,
        }

    def _load_llm_cache(self) -> dict:
        if self.llm_cache_path and os.path.exists(self.llm_cache_path):
            try:
                with open(self.llm_cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                print(f'[WARNING] Failed to load LLM checkpoint cache ({self.llm_cache_path}): {e}', flush=True)
        return {}

    def _save_llm_cache(self, cache_data: dict):
        if not self.llm_cache_path:
            return
        cache_dir = os.path.dirname(os.path.abspath(self.llm_cache_path))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = self.llm_cache_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.llm_cache_path)

    def _collect_llm_runs(
        self,
        provider_key: str,
        provider_label: str,
        transliterator,
        toponym_list: list,
        llm_cache: dict,
    ) -> list:
        """
        Execute K independent runs for a single LLM provider with batch-level disk checkpointing.
        Only queries uncached non-empty items per run and saves progress after every batch.
        """
        provider_cache = llm_cache.setdefault(provider_key, {})
        runs_output = []
        total_targets = len(toponym_list)

        for run_idx in range(1, self.num_llm_runs + 1):
            run_key = f'run_{run_idx}'
            run_map = provider_cache.setdefault(run_key, {})
            missing_targets = [t for t in toponym_list if not (run_map.get(t) and str(run_map.get(t)).strip())]

            if not missing_targets:
                print(
                    f'Step 3 [{provider_label}] (Run {run_idx}/{self.num_llm_runs}): '
                    f'Loaded 100% from checkpoint cache ({total_targets}/{total_targets}).',
                    flush=True,
                )
            elif self.use_batch:
                print(
                    f'Step 3 [{provider_label}] (Run {run_idx}/{self.num_llm_runs}): '
                    f'Fetching {len(missing_targets)}/{total_targets} uncached targets in BATCH mode '
                    f'(batch_size={self.batch_size})...',
                    flush=True,
                )
                for i in tqdm(
                    range(0, len(missing_targets), self.batch_size),
                    desc=f"{provider_label} Batches (Run {run_idx}/{self.num_llm_runs})",
                ):
                    chunk = missing_targets[i:i + self.batch_size]
                    chunk_res = transliterator.transliterate_batch(chunk)
                    for t, kana in chunk_res.items():
                        if kana and str(kana).strip():
                            run_map[t] = str(kana).strip()
                    self._save_llm_cache(llm_cache)
            else:
                print(
                    f'Step 3 [{provider_label}] (Run {run_idx}/{self.num_llm_runs}): '
                    f'Fetching {len(missing_targets)}/{total_targets} uncached targets in SEQUENTIAL mode...',
                    flush=True,
                )
                for t in tqdm(missing_targets, desc=f"{provider_label} Sequential (Run {run_idx}/{self.num_llm_runs})"):
                    kana = transliterator.transliterate(t)
                    if kana and str(kana).strip():
                        run_map[t] = str(kana).strip()
                    self._save_llm_cache(llm_cache)

            runs_output.append({t: run_map.get(t, '') for t in toponym_list})

        return runs_output

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f'Original total records: {len(df)}', flush=True)

        # 1. For all entries, determine rule-based transliteration based on Wikipedia pageviews threshold.
        print('Step 1: Applying Wikipedia pageviews threshold to rule-based transliteration...', flush=True)
        translit_a_list = []
        for _, row in df.iterrows():
            translit_a_list.append(self.rule_base_transliterator.transliterate(row))
        df['transliteration_a_kana'] = translit_a_list

        # 2. Making unique combination between Japanese transliteration and Ukrainian toponyms
        unique_df = df[['toponym', 'toponym_with_accent', 'transliteration_a_kana']].drop_duplicates(
            subset=['toponym_with_accent', 'transliteration_a_kana']
        ).copy().reset_index(drop=True)
        unique_df['eval_id'] = [f'eval_{i+1}' for i in range(len(unique_df))]
        print(f'Step 2: Deduplicated down to {len(unique_df)} unique evaluation combinations!', flush=True)

        # 3. Retrieving LLMs' transliterations across K independent runs (with disk checkpointing)
        accented_series = unique_df['toponym_with_accent'].fillna(unique_df['toponym']).astype(str).str.strip()
        toponym_list = list(dict.fromkeys(accented_series.tolist()))

        llm_cache = self._load_llm_cache()
        try:
            gpt_runs_per_run = self._collect_llm_runs(
                'gpt', 'Method B (GPT)', self.gpt_transliterator, toponym_list, llm_cache
            )
            gemini_runs_per_run = self._collect_llm_runs(
                'gemini', 'Method C (Gemini)', self.gemini_transliterator, toponym_list, llm_cache
            )
        except QuotaExhaustedError as qe:
            self._save_llm_cache(llm_cache)
            print(f'\n[PAUSED FOR API QUOTA / AUTH] {qe}', flush=True)
            if self.llm_cache_path:
                print(f'Checkpoint cache saved safely to: {self.llm_cache_path}', flush=True)
                for pk, pl in [('gpt', 'Method B (GPT)'), ('gemini', 'Method C (Gemini)')]:
                    status_str = ', '.join(
                        f'Run{r}={sum(1 for t in toponym_list if llm_cache.get(pk, {}).get(f"run_{r}", {}).get(t))}/{len(toponym_list)}'
                        for r in range(1, self.num_llm_runs + 1)
                    )
                    print(f'  {pl}: {status_str}', flush=True)
                print('Re-run the pipeline after quota resets to resume seamlessly without losing cached runs.', flush=True)
            raise

        # 4. G2P, TTS, FED, and Perplexity Evaluation with Memoization Cache
        print('Step 4: Running G2P, TTS, FED, and Perplexity Evaluations (with identical-transliteration caching)...')
        g2p_ppl_cache = {}     # kana_str -> (ipa_str, ppl_val)
        fed_cache = {}         # (cand_ipa, gt_ipa) -> fed_val
        audio_ja_cache = {}    # (e_id, method_tag, kana_str) -> wav_path
        acoustic_cache = {}    # (audio_gt, audio_cand) -> acoustic_dist

        def eval_kana_g2p_ppl(kana_str: str) -> tuple:
            if not kana_str or not str(kana_str).strip():
                return '', float('nan')
            clean_k = str(kana_str).strip()
            if clean_k not in g2p_ppl_cache:
                ipa_val = self.japanese_g2p.convert(clean_k)
                ppl_val = self.ppl.evaluate(clean_k)
                g2p_ppl_cache[clean_k] = (ipa_val, ppl_val)
            return g2p_ppl_cache[clean_k]

        def eval_fed_cached(cand_ipa: str, gt_ipa: str) -> float:
            if not cand_ipa or not gt_ipa:
                return float('nan')
            key = (cand_ipa, gt_ipa)
            if key not in fed_cache:
                fed_cache[key] = self.fed.evaluate(cand_ipa, gt_ipa)
            return fed_cache[key]

        def get_ja_audio_cached(kana_str: str, e_id: str, method_tag: str) -> str:
            if not kana_str or not str(kana_str).strip():
                return ''
            clean_k = str(kana_str).strip()
            key = (e_id, method_tag, clean_k)
            if key not in audio_ja_cache:
                audio_ja_cache[key] = self.audio_generator.generate_japanese(clean_k, e_id, method_tag)
            return audio_ja_cache[key]

        intermediate_rows = []
        all_audio_paths_set = set()

        for idx, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="G2P, TTS & Lexical Eval"):
            e_id = row['eval_id']
            ukr_word = str(row['toponym_with_accent']).strip() if pd.notna(row['toponym_with_accent']) else str(row['toponym']).strip()
            kana_a = str(row['transliteration_a_kana']).strip() if pd.notna(row['transliteration_a_kana']) else ''

            # Deterministic Ground Truth & Method A (computed once)
            gt_ipa = self.ukrainian_g2p.convert(ukr_word)
            ipa_a, ppl_a = eval_kana_g2p_ppl(kana_a)
            fed_a = eval_fed_cached(ipa_a, gt_ipa)

            audio_gt = self.audio_generator.generate_ukrainian(ukr_word, e_id)
            audio_a = get_ja_audio_cached(kana_a, e_id, 'method_a')
            if audio_gt:
                all_audio_paths_set.add(audio_gt)
            if audio_a:
                all_audio_paths_set.add(audio_a)

            # Multi-run Method B (GPT) and Method C (Gemini)
            b_kanas, b_ipas, b_feds, b_ppls, b_audios = [], [], [], [], []
            c_kanas, c_ipas, c_feds, c_ppls, c_audios = [], [], [], [], []

            for r_idx in range(self.num_llm_runs):
                kb = gpt_runs_per_run[r_idx].get(ukr_word, '')
                kc = gemini_runs_per_run[r_idx].get(ukr_word, '')

                ipab, pplb = eval_kana_g2p_ppl(kb)
                fedb = eval_fed_cached(ipab, gt_ipa)
                audb = get_ja_audio_cached(kb, e_id, 'method_b_gpt')
                if audb:
                    all_audio_paths_set.add(audb)

                ipac, pplc = eval_kana_g2p_ppl(kc)
                fedc = eval_fed_cached(ipac, gt_ipa)
                audc = get_ja_audio_cached(kc, e_id, 'method_c_gemini')
                if audc:
                    all_audio_paths_set.add(audc)

                b_kanas.append(kb)
                b_ipas.append(ipab)
                b_feds.append(fedb)
                b_ppls.append(pplb)
                b_audios.append(audb)

                c_kanas.append(kc)
                c_ipas.append(ipac)
                c_feds.append(fedc)
                c_ppls.append(pplc)
                c_audios.append(audc)

            intermediate_rows.append({
                'toponym_with_accent': row['toponym_with_accent'],
                'transliteration_a_kana': kana_a,
                'ground_truth_ipa': gt_ipa,
                'transliteration_a_ipa': ipa_a,
                'transliteration_a_fed': fed_a,
                'transliteration_a_ppl': ppl_a,
                'audio_gt': audio_gt,
                'audio_a': audio_a,
                'b_kanas': b_kanas,
                'b_ipas': b_ipas,
                'b_feds': b_feds,
                'b_ppls': b_ppls,
                'b_audios': b_audios,
                'c_kanas': c_kanas,
                'c_ipas': c_ipas,
                'c_feds': c_feds,
                'c_ppls': c_ppls,
                'c_audios': c_audios,
            })

        # Batched WavLM Layer 7 CMVN Feature Extraction over unique audio files
        unique_audio_paths = sorted(all_audio_paths_set)
        print(
            f'Calculating Acoustic Distances across {len(unique_audio_paths)} unique audio files '
            f'(cached from {len(unique_df) * (2 + 2 * self.num_llm_runs)} total evaluation slots)...'
        )
        if self.use_batch:
            feature_cache = self.acoustic.extract_features_batch(unique_audio_paths, batch_size=16)
        else:
            feature_cache = None

        def eval_acoustic_cached(aud_gt: str, aud_cand: str) -> float:
            if not aud_gt or not aud_cand:
                return float('nan')
            key = (aud_gt, aud_cand)
            if key not in acoustic_cache:
                acoustic_cache[key] = self.acoustic.evaluate(aud_gt, aud_cand, feature_cache)
            return acoustic_cache[key]

        eval_records = []
        for item in tqdm(intermediate_rows, desc="Acoustic DTW & Multi-Run Aggregation"):
            aud_gt = item['audio_gt']
            dist_a = eval_acoustic_cached(aud_gt, item['audio_a'])

            b_acoustics = [eval_acoustic_cached(aud_gt, aud_b) for aud_b in item['b_audios']]
            c_acoustics = [eval_acoustic_cached(aud_gt, aud_c) for aud_c in item['c_audios']]

            agg_b = self._aggregate_llm_runs(
                item['b_kanas'], item['b_ipas'], item['b_feds'], item['b_ppls'], b_acoustics, self.num_llm_runs
            )
            agg_c = self._aggregate_llm_runs(
                item['c_kanas'], item['c_ipas'], item['c_feds'], item['c_ppls'], c_acoustics, self.num_llm_runs
            )

            rec = {
                'toponym_with_accent': item['toponym_with_accent'],
                'transliteration_a_kana': item['transliteration_a_kana'],
                'ground_truth_ipa': item['ground_truth_ipa'],
                'transliteration_a_ipa': item['transliteration_a_ipa'],
                'transliteration_a_fed': item['transliteration_a_fed'],
                'transliteration_a_ppl': item['transliteration_a_ppl'],
                'transliteration_a_acoustic': dist_a,
                # Method B (5-run majority-vote string & 5-run mean metrics)
                'transliteration_b': agg_b['kana'],
                'transliteration_b_ipa': agg_b['ipa'],
                'transliteration_b_fed': agg_b['fed_mean'],
                'transliteration_b_ppl': agg_b['ppl_mean'],
                'transliteration_b_acoustic': agg_b['acoustic_mean'],
                'transliteration_b_fed_std': agg_b['fed_std'],
                'transliteration_b_ppl_std': agg_b['ppl_std'],
                'transliteration_b_acoustic_std': agg_b['acoustic_std'],
                'transliteration_b_num_unique': agg_b['num_unique'],
                'transliteration_b_agreement_rate': agg_b['agreement_rate'],
                'transliteration_b_exact_match': agg_b['exact_match'],
                # Method C (5-run majority-vote string & 5-run mean metrics)
                'transliteration_c': agg_c['kana'],
                'transliteration_c_ipa': agg_c['ipa'],
                'transliteration_c_fed': agg_c['fed_mean'],
                'transliteration_c_ppl': agg_c['ppl_mean'],
                'transliteration_c_acoustic': agg_c['acoustic_mean'],
                'transliteration_c_fed_std': agg_c['fed_std'],
                'transliteration_c_ppl_std': agg_c['ppl_std'],
                'transliteration_c_acoustic_std': agg_c['acoustic_std'],
                'transliteration_c_num_unique': agg_c['num_unique'],
                'transliteration_c_agreement_rate': agg_c['agreement_rate'],
                'transliteration_c_exact_match': agg_c['exact_match'],
            }

            # Store per-run trace columns (run1 .. runK)
            for r_idx in range(self.num_llm_runs):
                r_num = r_idx + 1
                rec[f'transliteration_b_run{r_num}'] = item['b_kanas'][r_idx]
                rec[f'transliteration_b_fed_run{r_num}'] = item['b_feds'][r_idx]
                rec[f'transliteration_b_ppl_run{r_num}'] = item['b_ppls'][r_idx]
                rec[f'transliteration_b_acoustic_run{r_num}'] = b_acoustics[r_idx]

                rec[f'transliteration_c_run{r_num}'] = item['c_kanas'][r_idx]
                rec[f'transliteration_c_fed_run{r_num}'] = item['c_feds'][r_idx]
                rec[f'transliteration_c_ppl_run{r_num}'] = item['c_ppls'][r_idx]
                rec[f'transliteration_c_acoustic_run{r_num}'] = c_acoustics[r_idx]

            eval_records.append(rec)

        eval_df = pd.DataFrame(eval_records)

        # Print multi-run self-consistency summary
        print(f'\n--- Multi-Run LLM Reproducibility Summary (K = {self.num_llm_runs} runs, N = {len(eval_df)}) ---')
        for m_code, m_label in [('b', 'Method B (GPT-4o-mini)'), ('c', 'Method C (Gemini-3.1-FL)')]:
            exact_pct = eval_df[f'transliteration_{m_code}_exact_match'].mean() * 100.0
            agree_pct = eval_df[f'transliteration_{m_code}_agreement_rate'].mean() * 100.0
            mean_fed_sd = eval_df[f'transliteration_{m_code}_fed_std'].mean()
            mean_ppl_sd = eval_df[f'transliteration_{m_code}_ppl_std'].mean()
            mean_ac_sd = eval_df[f'transliteration_{m_code}_acoustic_std'].mean()
            print(
                f'  {m_label:26s}: Exact {self.num_llm_runs}/{self.num_llm_runs} Match = {exact_pct:6.2f}% | '
                f'Mean Agreement = {agree_pct:6.2f}% | '
                f'Mean Within-Toponym SD: FED={mean_fed_sd:.4f}, PPL={mean_ppl_sd:.4f}, Acoustic={mean_ac_sd:.4f}'
            )

        # 5. Merge the results back into the original dataframe to achieve a complete reconstruction
        print('\nStep 5: Merging evaluated unique records back into the full dataset...')
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

    parser = argparse.ArgumentParser(description='Ukrainian Toponym Transliteration Multi-Dimensional Evaluation Pipeline')
    parser.add_argument('--mock', action='store_true', help='Run mock test on the first 5 records')
    parser.add_argument('--num-llm-runs', type=int, default=5, help='Number of LLM runs to average (default: 5)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for LLM API calls (default: 60 production, 5 mock)')
    parser.add_argument('--llm-cache-path', type=str, default=None, help='Path to persistent JSON checkpoint cache for LLM runs')
    parser.add_argument('--clear-llm-cache', action='store_true', help='Remove existing LLM checkpoint cache before running')
    args, _ = parser.parse_known_args()

    is_mock = MOCK_MODE or args.mock
    num_runs = args.num_llm_runs
    batch_sz = args.batch_size if args.batch_size is not None else (5 if is_mock else 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    results_dir = os.path.join(project_root, 'results')
    audio_dir = os.path.join(project_root, 'audio')
    os.makedirs(results_dir, exist_ok=True)

    csv_candidates = [
        os.path.join(project_root, 'data', 'processed', 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'),
        os.path.join(project_root, 'data', 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'),
        'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'
    ]
    csv_file = next((p for p in csv_candidates if os.path.exists(p)), csv_candidates[0])
    if not os.path.exists(csv_file):
        results_dir = '.'

    llm_cache_path = args.llm_cache_path
    if llm_cache_path is None and not is_mock:
        llm_cache_path = os.path.join(results_dir, 'llm_run_cache.json')
    if args.clear_llm_cache and llm_cache_path and os.path.exists(llm_cache_path):
        os.remove(llm_cache_path)
        print(f"Cleared existing LLM checkpoint cache: {llm_cache_path}", flush=True)

    print(f"Loading data from {csv_file}...", flush=True)
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
        batch_size=batch_sz,
        num_llm_runs=num_runs,
        base_audio_dir=audio_dir,
        llm_cache_path=llm_cache_path,
    )

    # 3. If conditions depending on MOCK_MODE / --mock boolean value
    if is_mock:
        print(f"\n[MOCK MODE] Testing pipeline with the first 5 records only (num_llm_runs={num_runs})...", flush=True)
        eval_subset = df.head(5).copy()
        output_csv = os.path.join(results_dir, 'mock_evaluation_results.csv')
    else:
        print(f"\n[PRODUCTION MODE] Running full pipeline for all {len(df)} records (num_llm_runs={num_runs}, batch_size={batch_sz})...", flush=True)
        eval_subset = df.copy()
        output_csv = os.path.join(results_dir, 'ukrainian_toponyms_evaluation_results.csv')

    # 4. Executing pipeline
    try:
        df_evaluated = pipeline.run_pipeline(eval_subset)
    except QuotaExhaustedError:
        sys.exit(1)

    # 5. Saving the result
    df_evaluated.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nCompleted! Results saved successfully to: {output_csv}", flush=True)
