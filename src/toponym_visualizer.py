import os
import re
import pandas as pd
import numpy as np
import warnings
import altair as alt
warnings.filterwarnings('ignore', message='.*Automatically deduplicated selection parameter.*')

## Toponym Evaluation Visualizer using Altair
class ToponymVisualizer:

    def __init__(self, theme='latimes'):
        try:
            alt.theme.enable(theme)
        except Exception:
            pass

    # Cleansing parentheses for disambiguation page on Wikipedia
    def clean_title(self, title):
        if pd.isna(title):
            return ''
        return re.sub(r'[\s　]*[\(（].*?[\)）]', '', str(title)).strip()

    # 1. Reshape wide evaluation DataFrame into long format for Altair multi-method comparisons
    def _prepare_long_format(self, df):
        records = []
        methods = [
            ('Method A (Rule-Based)', 'transliteration_a_kana', 'transliteration_a_fed', 'transliteration_a_ppl', 'transliteration_a_acoustic'),
            ('Method B (GPT-4o-mini)', 'transliteration_b', 'transliteration_b_fed', 'transliteration_b_ppl', 'transliteration_b_acoustic'),
            ('Method C (Gemini-3.5-Flash)', 'transliteration_c', 'transliteration_c_fed', 'transliteration_c_ppl', 'transliteration_c_acoustic'),
        ]
        
        for _, r in df.iterrows():
            ukr = r.get('toponym', '')
            ukr_accent = r.get('toponym_with_accent', ukr)
            pv = r.get('avg_monthly_pv_2y', 0)
            if pd.isna(pv):
                pv = 0
            
            for m_label, col_kana, col_fed, col_ppl, col_ac in methods:
                kana = r.get(col_kana, '')
                fed = r.get(col_fed, np.nan)
                ppl = r.get(col_ppl, np.nan)
                acoustic = r.get(col_ac, np.nan)
                
                if pd.notna(kana) and str(kana).strip() != '':
                    fed_val = float(fed) if (pd.notna(fed) and np.isfinite(float(fed))) else np.nan
                    ppl_val = float(ppl) if (pd.notna(ppl) and np.isfinite(float(ppl))) else np.nan
                    ac_val = float(acoustic) if (pd.notna(acoustic) and np.isfinite(float(acoustic))) else np.nan
                    records.append({
                        'Ukrainian Toponym': ukr,
                        'Toponym (with Accent)': ukr_accent,
                        'Method': m_label,
                        'Transliteration': str(kana).strip(),
                        'FED (Phonetic Distance)': fed_val,
                        'Perplexity (PPL)': ppl_val,
                        'Acoustic Distance': ac_val,
                        'Avg Monthly Pageviews': float(pv)
                    })
        return pd.DataFrame(records)

    # 2. FED (Feature Edit Distance) Distribution Graphic
    def plot_fed_distribution(self, df_long):
        valid = df_long.dropna(subset=['FED (Phonetic Distance)']).copy()
        if len(valid) == 0:
            return alt.Chart().mark_text(text="No valid FED data available.")

        chart = alt.Chart(valid).mark_boxplot(extent='min-max', size=42).encode(
            x=alt.X('Method:N', title='Transliteration Method', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('FED (Phonetic Distance):Q', title='Feature Edit Distance (Lower = Closer to Ukrainian IPA)'),
            color=alt.Color('Method:N', scale=alt.Scale(scheme='tableau10'), legend=None),
            tooltip=[
                'Method:N',
                alt.Tooltip('min(FED (Phonetic Distance)):Q', title='Min', format='.3f'),
                alt.Tooltip('q1(FED (Phonetic Distance)):Q', title='Q1', format='.3f'),
                alt.Tooltip('median(FED (Phonetic Distance)):Q', title='Median', format='.3f'),
                alt.Tooltip('q3(FED (Phonetic Distance)):Q', title='Q3', format='.3f'),
                alt.Tooltip('max(FED (Phonetic Distance)):Q', title='Max', format='.3f')
            ]
        ).properties(
            title={
                'text': 'Phonological Fidelity: Feature Edit Distance (FED)',
                'subtitle': 'Subsegmental articulatory feature distance via PanPhon against Ukrainian IPA'
            },
            width=360,
            height=320
        )
        return chart

    # 3. Acoustic Distance Distribution Graphic (Meta MMS-1B Layer 7 + FastDTW)
    def plot_acoustic_distribution(self, df_long):
        valid = df_long.dropna(subset=['Acoustic Distance']).copy()
        if len(valid) == 0:
            return alt.Chart().mark_text(text="No valid Acoustic Distance data available.")

        chart = alt.Chart(valid).mark_boxplot(extent='min-max', size=42).encode(
            x=alt.X('Method:N', title='Transliteration Method', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Acoustic Distance:Q', title='Normalized Acoustic Distance (Lower = Closer to Original Spoken Audio)'),
            color=alt.Color('Method:N', scale=alt.Scale(scheme='tableau10'), legend=None),
            tooltip=[
                'Method:N',
                alt.Tooltip('min(Acoustic Distance):Q', title='Min', format='.4f'),
                alt.Tooltip('q1(Acoustic Distance):Q', title='Q1', format='.4f'),
                alt.Tooltip('median(Acoustic Distance):Q', title='Median', format='.4f'),
                alt.Tooltip('q3(Acoustic Distance):Q', title='Q3', format='.4f'),
                alt.Tooltip('max(Acoustic Distance):Q', title='Max', format='.4f')
            ]
        ).properties(
            title={
                'text': 'Acoustic Fidelity: Meta MMS-1B (Layer 7) + FastDTW',
                'subtitle': 'Dynamic Time Warping distance between synthesized Ukrainian and Japanese audio'
            },
            width=360,
            height=320
        )
        return chart

    # 4. Perplexity (PPL) Distribution Graphic (Phonotactic Naturalness)
    def plot_ppl_distribution(self, df_long):
        valid = df_long.dropna(subset=['Perplexity (PPL)']).copy()
        if len(valid) == 0:
            return alt.Chart().mark_text(text="No valid Perplexity data available.")

        chart = alt.Chart(valid).mark_boxplot(extent='min-max', size=42).encode(
            x=alt.X('Method:N', title='Transliteration Method', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Perplexity (PPL):Q', title='Perplexity (Lower = More Natural Japanese Phonotactics)'),
            color=alt.Color('Method:N', scale=alt.Scale(scheme='tableau10'), legend=None),
            tooltip=[
                'Method:N',
                alt.Tooltip('min(Perplexity (PPL)):Q', title='Min', format='.2f'),
                alt.Tooltip('q1(Perplexity (PPL)):Q', title='Q1', format='.2f'),
                alt.Tooltip('median(Perplexity (PPL)):Q', title='Median', format='.2f'),
                alt.Tooltip('q3(Perplexity (PPL)):Q', title='Q3', format='.2f'),
                alt.Tooltip('max(Perplexity (PPL)):Q', title='Max', format='.2f')
            ]
        ).properties(
            title={
                'text': 'Phonotactic Naturalness: N-gram Perplexity (PPL)',
                'subtitle': 'Kneser-Ney 3-gram language model trained on UniDic-lite Katakana vocabulary'
            },
            width=360,
            height=320
        )
        return chart

    # 5. Multi-Method Summary Bar Chart (Mean & Std Dev across FED, Acoustic, PPL)
    def plot_methods_summary(self, df_long):
        metrics = [
            ('FED (Phonetic Distance)', 'Phonological (FED)'),
            ('Acoustic Distance', 'Acoustic (MMS-1B)'),
            ('Perplexity (PPL)', 'Phonotactic (PPL)')
        ]
        records = []
        for col_name, label in metrics:
            sub = df_long.dropna(subset=[col_name]).copy()
            for method, grp in sub.groupby('Method'):
                records.append({
                    'Method': method,
                    'Metric': label,
                    'Mean': float(grp[col_name].mean()),
                    'Median': float(grp[col_name].median()),
                    'Std': float(grp[col_name].std()),
                    'Count': len(grp)
                })
        sum_df = pd.DataFrame(records)
        if len(sum_df) == 0:
            return alt.Chart().mark_text(text="No data for summary.")

        chart = alt.Chart(sum_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Method:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y('Mean:Q', title='Mean Metric Value'),
            color=alt.Color('Method:N', scale=alt.Scale(scheme='tableau10'), title='Transliteration Method'),
            column=alt.Column('Metric:N', title='Comparative Metric Overview across Methods', header=alt.Header(labelFontSize=12, titleFontSize=13)),
            tooltip=[
                'Method:N',
                'Metric:N',
                alt.Tooltip('Mean:Q', format='.3f'),
                alt.Tooltip('Median:Q', format='.3f'),
                alt.Tooltip('Std:Q', format='.3f'),
                alt.Tooltip('Count:Q', format=',')
            ]
        ).properties(
            width=180,
            height=280
        )
        return chart

    # 6. Interactive Scatter Plot: Acoustic Distance vs. Perplexity (Trade-off)
    def plot_tradeoff_scatter(self, df_long):
        valid_long = df_long.dropna(subset=['Acoustic Distance', 'Perplexity (PPL)']).copy()
        if len(valid_long) == 0:
            return alt.Chart().mark_text(text="No evaluation data available yet.")

        # Interactive selection for legend filtering
        selection = alt.selection_point(name='method_sel', fields=['Method'], bind='legend')

        chart = alt.Chart(valid_long).mark_circle(size=75, opacity=0.75).encode(
            x=alt.X('Acoustic Distance:Q', title='Normalized Acoustic Distance (Lower = Closer Spoken Match)', scale=alt.Scale(zero=False)),
            y=alt.Y('Perplexity (PPL):Q', title='Phonotactic Complexity (PPL, Lower = More Natural Japanese)', scale=alt.Scale(zero=False)),
            color=alt.Color('Method:N', scale=alt.Scale(scheme='tableau10'), title='Transliteration Method'),
            opacity=alt.condition(selection, alt.value(0.85), alt.value(0.12)),
            tooltip=[
                'Ukrainian Toponym:N',
                'Method:N',
                'Transliteration:N',
                alt.Tooltip('FED (Phonetic Distance):Q', format='.3f'),
                alt.Tooltip('Perplexity (PPL):Q', format='.2f'),
                alt.Tooltip('Acoustic Distance:Q', format='.4f'),
                alt.Tooltip('Avg Monthly Pageviews:Q', format=',.0f')
            ]
        ).properties(
            title={
                'text': 'Acoustic Fidelity vs. Phonotactic Naturalness Trade-off',
                'subtitle': 'Hover over points to inspect individual toponyms. Click legend items to filter methods.'
            },
            width=680,
            height=440
        ).add_params(
            selection
        ).interactive()

        return chart

    # 7. Interactive Knee Point and Pageview Curve
    def plot_interactive_knee(self, df, knee_pv=533.8, knee_rank=None):
        valid_df = df[df['avg_monthly_pv_2y'].notna()].copy()
        valid_df['clean_ja_wiki'] = valid_df['ja_wiki_title'].apply(self.clean_title)
        valid_df = valid_df.sort_values(by='avg_monthly_pv_2y', ascending=False).reset_index(drop=True)
        valid_df['Rank'] = np.arange(1, len(valid_df) + 1)
        valid_df['Is Established'] = (valid_df['avg_monthly_pv_2y'] >= knee_pv) & (valid_df['has_ja_wiki'] == True) & (valid_df['clean_ja_wiki'] != '')
        if knee_rank is None or knee_rank > len(valid_df):
            knee_rank = max(1, int((valid_df['avg_monthly_pv_2y'] >= knee_pv).sum()))

        plot_df = valid_df[['Rank', 'avg_monthly_pv_2y', 'toponym', 'clean_ja_wiki', 'Is Established']].copy()

        # Main Pageviews Curve
        line = alt.Chart(plot_df).mark_line(color='#1f77b4', size=2).encode(
            x=alt.X('Rank:Q', title='Toponym Rank (Sorted by Pageviews in desc. order)'),
            y=alt.Y('avg_monthly_pv_2y:Q', title='Avg. Monthly Pageviews (2-Year Average)')
        )

        # Established Toponyms Highlighted as Red Dots
        established_points = alt.Chart(plot_df[plot_df['Is Established']]).mark_circle(size=60, color='#d62728', opacity=0.85).encode(
            x='Rank:Q',
            y='avg_monthly_pv_2y:Q',
            tooltip=[
                alt.Tooltip('Rank:Q', title='Rank'),
                alt.Tooltip('toponym:N', title='Ukrainian Toponym'),
                alt.Tooltip('clean_ja_wiki:N', title='Established Japanese Title'),
                alt.Tooltip('avg_monthly_pv_2y:Q', title='Monthly Pageviews', format=',.0f')
            ]
        )

        # Horizontal Threshold Line (Knee Pageviews)
        h_rule = alt.Chart(pd.DataFrame({'knee_y': [knee_pv]})).mark_rule(
            color='#d62728', strokeDash=[4, 4], size=1.5
        ).encode(y='knee_y:Q')

        # Vertical Knee Rank Line
        v_rule = alt.Chart(pd.DataFrame({'knee_x': [knee_rank]})).mark_rule(
            color='#d62728', strokeDash=[4, 4], size=1.5
        ).encode(x='knee_x:Q')

        # Knee Point Highlight
        knee_point = alt.Chart(pd.DataFrame({'knee_x': [knee_rank], 'knee_y': [knee_pv]})).mark_circle(
            size=120, color='#d62728'
        ).encode(
            x='knee_x:Q',
            y='knee_y:Q',
            tooltip=[
                alt.Tooltip('knee_x:Q', title='Knee Rank'),
                alt.Tooltip('knee_y:Q', title='Threshold (views/month)', format='.1f')
            ]
        )

        chart = (line + established_points + h_rule + v_rule + knee_point).properties(
            title={
                'text': 'Wikipedia Pageview Distribution & Knee Point Detection (Interactive)',
                'subtitle': f'Threshold: {knee_pv:.1f} views/month (Rank {knee_rank}). Red dots: Established toponyms ({plot_df["Is Established"].sum()} items).'
            },
            width=700,
            height=400
        ).interactive()

        return chart

    # 8. Build full unified HTML dashboard
    def build_dashboard(self, df_evaluated, knee_pv=533.8, knee_rank=500, output_html='evaluation_dashboard.html'):
        df_long = self._prepare_long_format(df_evaluated)
        
        # Build individual plots
        knee_plot = self.plot_interactive_knee(df_evaluated, knee_pv, knee_rank)
        scatter = self.plot_tradeoff_scatter(df_long)
        box_fed = self.plot_fed_distribution(df_long)
        box_ac = self.plot_acoustic_distribution(df_long)
        box_ppl = self.plot_ppl_distribution(df_long)
        summary_bar = self.plot_methods_summary(df_long)

        # Concatenate layout:
        # Panel 1: Wikipedia Pageviews Knee Curve
        # Panel 2: Acoustic Fidelity vs. Phonotactic Naturalness Trade-off Scatter
        # Panel 3: 3-Way Metric Distributions Side-by-Side (FED | Acoustic | PPL)
        # Panel 4: Overall Method Comparison Summary Bar Chart
        dashboard = alt.vconcat(
            knee_plot,
            scatter,
            alt.hconcat(box_fed, box_ac, box_ppl).resolve_scale(color='independent'),
            summary_bar.resolve_scale(color='independent')
        ).resolve_scale(
            color='independent'
        )

        dashboard.save(output_html)
        print(f"Unified Interactive Altair dashboard saved successfully to: {output_html}")
        return output_html

    # 9. Export all individual presentation graphics as standalone HTML files
    def export_all_graphics(self, df_evaluated, knee_pv=533.8, knee_rank=500, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)
        df_long = self._prepare_long_format(df_evaluated)

        files_saved = {}

        # 1. FED Distribution
        p_fed = os.path.join(output_dir, 'graphics_fed_distribution.html')
        self.plot_fed_distribution(df_long).save(p_fed)
        files_saved['FED Distribution'] = p_fed

        # 2. Acoustic Distance Distribution
        p_ac = os.path.join(output_dir, 'graphics_acoustic_distribution.html')
        self.plot_acoustic_distribution(df_long).save(p_ac)
        files_saved['Acoustic Distribution'] = p_ac

        # 3. PPL Distribution
        p_ppl = os.path.join(output_dir, 'graphics_ppl_distribution.html')
        self.plot_ppl_distribution(df_long).save(p_ppl)
        files_saved['PPL Distribution'] = p_ppl

        # 4. Summary Bar Chart
        p_sum = os.path.join(output_dir, 'graphics_methods_summary.html')
        self.plot_methods_summary(df_long).save(p_sum)
        files_saved['Methods Summary'] = p_sum

        # 5. Trade-off Scatter
        p_trade = os.path.join(output_dir, 'graphics_tradeoff_scatter.html')
        self.plot_tradeoff_scatter(df_long).save(p_trade)
        files_saved['Trade-off Scatter'] = p_trade

        # 6. Wikipedia Pageview Interactive Curve
        p_knee = os.path.join(output_dir, 'wikipedia_pageview_interactive.html')
        self.plot_interactive_knee(df_evaluated, knee_pv, knee_rank).save(p_knee)
        files_saved['Knee Curve'] = p_knee

        # 7. Unified Comprehensive Dashboard
        p_dash = os.path.join(output_dir, 'evaluation_dashboard.html')
        self.build_dashboard(df_evaluated, knee_pv, knee_rank, output_html=p_dash)
        files_saved['Unified Dashboard'] = p_dash

        print(f"\nAll {len(files_saved)} visualization graphics exported successfully to {output_dir}/:")
        for name, path in files_saved.items():
            print(f"  - {name}: {path}")

        return files_saved

    # 10. Generate mock dataset for self-testing and pipeline verification
    @staticmethod
    def get_mock_dataframe():
        sample_data = {
            'toponym': ['Київ', 'Львів', 'Одеса', 'Харків', 'Дніпро', 'Запоріжжя', 'Вінниця', 'Полтава', 'Чернігів', 'Ужгород'],
            'toponym_with_accent': ['Ки́їв', 'Львів', 'Оде́са', 'Ха́рків', 'Дніпро́', 'Запорі́жжя', 'Ві́нниця', 'Полта́ва', 'Черні́гів', 'У́жгород'],
            'avg_monthly_pv_2y': [125000, 48000, 32000, 29000, 18000, 14000, 8500, 6200, 4500, 2800],
            'has_ja_wiki': [True]*10,
            'ja_wiki_title': ['キーウ', 'リヴィウ', 'オデッサ', 'ハルキウ', 'ドニプロ', 'ザポリージャ', 'ヴィーンヌィツャ', 'ポルタヴァ', 'チェルニーヒウ', 'ウージュホロド'],
            'transliteration_a_kana': ['キーウ', 'リヴィウ', 'オデッサ', 'ハルキウ', 'ドニプロ', 'ザポリージャ', 'ヴィーンヌィツャ', 'ポルタヴァ', 'チェルニーヒウ', 'ウージュホロド'],
            'transliteration_a_fed': [0.12, 0.18, 0.15, 0.14, 0.22, 0.19, 0.25, 0.16, 0.20, 0.21],
            'transliteration_a_ppl': [24.5, 31.2, 18.4, 28.1, 35.0, 38.6, 42.1, 26.3, 33.7, 36.2],
            'transliteration_a_acoustic': [0.185, 0.210, 0.195, 0.205, 0.240, 0.225, 0.260, 0.198, 0.230, 0.245],
            'transliteration_b': ['キーウ', 'リヴィウ', 'オデーサ', 'ハルキウ', 'ドニプロ', 'ザポリージャ', 'ヴィンニツァ', 'ポルタヴァ', 'チェルニヒフ', 'ウジホロド'],
            'transliteration_b_fed': [0.12, 0.18, 0.16, 0.14, 0.22, 0.19, 0.21, 0.16, 0.24, 0.26],
            'transliteration_b_ppl': [24.5, 31.2, 22.0, 28.1, 35.0, 38.6, 32.5, 26.3, 29.4, 30.1],
            'transliteration_b_acoustic': [0.185, 0.210, 0.202, 0.205, 0.240, 0.225, 0.235, 0.198, 0.250, 0.270],
            'transliteration_c': ['キエフ', 'リビウ', 'オデッサ', 'ハリコフ', 'ドニエプル', 'ザポロジエ', 'ビンニツァ', 'ポルタバ', 'チェルニゴフ', 'ウジゴロド'],
            'transliteration_c_fed': [0.28, 0.32, 0.15, 0.35, 0.38, 0.34, 0.36, 0.29, 0.37, 0.39],
            'transliteration_c_ppl': [15.2, 20.4, 18.4, 19.8, 22.5, 25.1, 21.0, 17.5, 20.2, 22.1],
            'transliteration_c_acoustic': [0.310, 0.345, 0.195, 0.380, 0.395, 0.360, 0.375, 0.305, 0.385, 0.410]
        }
        return pd.DataFrame(sample_data)

    # 11. Run complete mock test suite validating all Altair visual components
    def run_mock_test(self, output_dir='results/mock_visualizations'):
        os.makedirs(output_dir, exist_ok=True)
        print('=' * 60)
        print('  [MOCK TEST] ToponymVisualizer Altair Test Suite')
        print('=' * 60)
        
        mock_df = self.get_mock_dataframe()
        print(f'Generated synthetic evaluation dataset ({len(mock_df)} toponyms, 3 methods).')
        print(f'Target mock export directory: {os.path.abspath(output_dir)}\n')

        # Run full export suite
        saved_files = self.export_all_graphics(mock_df, knee_pv=533.8, knee_rank=5, output_dir=output_dir)

        # Verification step
        print('\nVerifying exported HTML artifacts...')
        all_passed = True
        for name, path in saved_files.items():
            if not os.path.exists(path):
                print(f'  [FAIL] {name}: File not found ({path})')
                all_passed = False
                continue
            
            size_kb = os.path.getsize(path) / 1024.0
            if size_kb < 1.0:
                print(f'  [FAIL] {name}: File is unexpectedly small ({size_kb:.1f} KB)')
                all_passed = False
                continue

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read(2048)
                if 'vega' not in content.lower() and 'altair' not in content.lower() and '<!DOCTYPE html>' not in content:
                    print(f'  [FAIL] {name}: File does not appear to be valid Vega/Altair HTML')
                    all_passed = False
                    continue

            print(f'  [PASS] {name:28s} ({size_kb:6.1f} KB) -> {os.path.basename(path)}')

        print('=' * 60)
        if all_passed:
            print('  SUCCESS: All 7 Altair visualization mock tests passed!')
        else:
            print('  FAILURE: One or more visualization mock tests failed!')
        print('=' * 60)
        return all_passed

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ukrainian Toponym Evaluation Visualizer (Altair)')
    parser.add_argument('--mock', action='store_true', help='Run mock evaluation test with synthetic data')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory for exported HTML files')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    eval_csv = os.path.join(results_dir, 'ukrainian_toponyms_evaluation_results.csv')
    mock_csv = os.path.join(results_dir, 'mock_evaluation_results.csv')
    
    visualizer = ToponymVisualizer()

    if args.mock:
        out_dir = args.output_dir if args.output_dir else os.path.join(results_dir, 'mock_visualizations')
        visualizer.run_mock_test(output_dir=out_dir)
    elif os.path.exists(eval_csv):
        out_dir = args.output_dir if args.output_dir else results_dir
        print(f'Loading evaluated production results from: {eval_csv}')
        df = pd.read_csv(eval_csv)
        visualizer.export_all_graphics(df, knee_pv=533.8, knee_rank=500, output_dir=out_dir)
    elif os.path.exists(mock_csv):
        out_dir = args.output_dir if args.output_dir else results_dir
        print(f'Loading mock evaluation results from: {mock_csv}')
        df = pd.read_csv(mock_csv)
        visualizer.export_all_graphics(df, knee_pv=533.8, knee_rank=500, output_dir=out_dir)
    else:
        out_dir = args.output_dir if args.output_dir else os.path.join(results_dir, 'mock_visualizations')
        print('No evaluation CSV found. Automatically executing mock evaluation test...')
        visualizer.run_mock_test(output_dir=out_dir)
