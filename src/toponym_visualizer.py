import os
import re
import pandas as pd
import numpy as np
import altair as alt

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
                    records.append({
                        'Ukrainian Toponym': ukr,
                        'Toponym (with Accent)': ukr_accent,
                        'Method': m_label,
                        'Transliteration': kana,
                        'FED (Phonetic Distance)': float(fed) if pd.notna(fed) else np.nan,
                        'Perplexity (PPL)': float(ppl) if pd.notna(ppl) else np.nan,
                        'Acoustic Distance': float(acoustic) if pd.notna(acoustic) else np.nan,
                        'Avg Monthly Pageviews': float(pv)
                    })
        return pd.DataFrame(records)

    # 2. Interactive Scatter Plot: Acoustic Distance vs. Perplexity
    def plot_tradeoff_scatter(self, df_long):
        # Filter rows with valid evaluation numbers
        valid_long = df_long.dropna(subset=['Acoustic Distance', 'Perplexity (PPL)']).copy()
        if len(valid_long) == 0:
            return alt.Chart().mark_text(text="No evaluation data available yet.")

        # Interactive selection for legend filtering
        selection = alt.selection_point(fields=['Method'], bind='legend')

        chart = alt.Chart(valid_long).mark_circle(size=80, opacity=0.7).encode(
            x=alt.X('Acoustic Distance:Q', title='Normalized Acoustic Distance (Lower = Closer to Original Sound)', scale=alt.Scale(zero=False)),
            y=alt.Y('Perplexity (PPL):Q', title='Phonotactic Complexity (PPL, Lower = More Natural Japanese)', scale=alt.Scale(zero=False)),
            color=alt.Color('Method:N', scale=alt.Scale(scheme='category10'), title='Transliteration Method'),
            opacity=alt.condition(selection, alt.value(0.85), alt.value(0.1)),
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
            width=650,
            height=450
        ).add_params(
            selection
        ).interactive()

        return chart

    # 3. Grouped Boxplots comparing metric distributions across methods
    def plot_metric_distributions(self, df_long, metric_col='FED (Phonetic Distance)'):
        valid_long = df_long.dropna(subset=[metric_col]).copy()
        if len(valid_long) == 0:
            return alt.Chart().mark_text(text=f"No data for {metric_col}")

        chart = alt.Chart(valid_long).mark_boxplot(extent='min-max', size=40).encode(
            x=alt.X('Method:N', title='Transliteration Method', axis=alt.Axis(labelAngle=-15)),
            y=alt.Y(f'{metric_col}:Q', title=metric_col),
            color=alt.Color('Method:N', legend=None)
        ).properties(
            title=f'Distribution Comparison of {metric_col} Across Methods',
            width=380,
            height=320
        )
        return chart

    # 4. Interactive Knee Point and Pageview Curve (Fixed Linear Scale & Clean Layout)
    def plot_interactive_knee(self, df, knee_pv, knee_rank=500):
        valid_df = df[df['avg_monthly_pv_2y'].notna()].copy()
        valid_df['clean_ja_wiki'] = valid_df['ja_wiki_title'].apply(self.clean_title)
        valid_df = valid_df.sort_values(by='avg_monthly_pv_2y', ascending=False).reset_index(drop=True)
        valid_df['Rank'] = np.arange(1, len(valid_df) + 1)
        valid_df['Is Established'] = (valid_df['avg_monthly_pv_2y'] >= knee_pv) & (valid_df['has_ja_wiki'] == True) & (valid_df['clean_ja_wiki'] != '')

        # Select strictly necessary columns to keep JSON payload compact
        plot_df = valid_df[['Rank', 'avg_monthly_pv_2y', 'toponym', 'clean_ja_wiki', 'Is Established']].copy()

        # 1. Main Pageviews Curve (Linear Scale matching Matplotlib)
        line = alt.Chart(plot_df).mark_line(color='#1f77b4', size=2).encode(
            x=alt.X('Rank:Q', title='Toponym Rank (Sorted by Pageview in desc. order)'),
            y=alt.Y('avg_monthly_pv_2y:Q', title='Avg. Monthly Pageviews (2-Year Average)')
        )

        # 2. Established Toponyms Highlighted as Red Dots
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

        # 3. Horizontal Threshold Line (Knee Pageviews)
        h_rule = alt.Chart(pd.DataFrame({'knee_y': [knee_pv]})).mark_rule(
            color='#d62728', strokeDash=[4, 4], size=1.5
        ).encode(
            y='knee_y:Q'
        )

        # 4. Vertical Knee Rank Line
        v_rule = alt.Chart(pd.DataFrame({'knee_x': [knee_rank]})).mark_rule(
            color='#d62728', strokeDash=[4, 4], size=1.5
        ).encode(
            x='knee_x:Q'
        )

        # 5. Knee Point Highlight
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
                'subtitle': f'Horizontal line: Threshold ({knee_pv:.1f} views/month). Red circles: Established toponyms ({plot_df["Is Established"].sum()} items).'
            },
            width=700,
            height=420
        ).interactive()

        return chart

    # 5. Build full HTML dashboard
    def build_dashboard(self, df_evaluated, knee_pv, knee_rank=500, output_html='evaluation_dashboard.html'):
        df_long = self._prepare_long_format(df_evaluated)
        
        # Build individual plots
        scatter = self.plot_tradeoff_scatter(df_long)
        box_fed = self.plot_metric_distributions(df_long, 'FED (Phonetic Distance)')
        box_ppl = self.plot_metric_distributions(df_long, 'Perplexity (PPL)')
        knee_plot = self.plot_interactive_knee(df_evaluated, knee_pv, knee_rank)

        # Concatenate layout:
        # Top: Knee plot
        # Middle: Trade-off scatter
        # Bottom: Side-by-side boxplots
        dashboard = alt.vconcat(
            knee_plot,
            scatter,
            alt.hconcat(box_fed, box_ppl)
        ).resolve_scale(
            color='independent'
        )

        dashboard.save(output_html)
        print(f"Interactive Altair dashboard saved successfully to: {output_html}")
        return output_html

if __name__ == '__main__':
    import sys

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    results_dir = os.path.join(project_root, 'results')
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if evaluated results CSV is available, otherwise generate interactive knee plot from raw dataset
    eval_csv = os.path.join(results_dir, 'ukrainian_toponyms_evaluation_results.csv')
    mock_csv = os.path.join(results_dir, 'mock_evaluation_results.csv')
    raw_candidates = [
        os.path.join(data_dir, 'processed', 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'),
        os.path.join(data_dir, 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'),
        'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'
    ]
    raw_csv = next((p for p in raw_candidates if os.path.exists(p)), raw_candidates[0])
    
    visualizer = ToponymVisualizer()
    dashboard_html = os.path.join(results_dir, 'evaluation_dashboard.html')
    knee_html = os.path.join(results_dir, 'wikipedia_pageview_interactive.html')
    
    if os.path.exists(eval_csv):
        print(f"Loading evaluated results from: {eval_csv}")
        df = pd.read_csv(eval_csv)
        visualizer.build_dashboard(df, knee_pv=534.0, knee_rank=500, output_html=dashboard_html)
    elif os.path.exists(mock_csv):
        print(f"Loading mock evaluation results from: {mock_csv}")
        df = pd.read_csv(mock_csv)
        visualizer.build_dashboard(df, knee_pv=534.0, knee_rank=500, output_html=dashboard_html)
    else:
        print(f"Evaluated pipeline data not yet detected. Generating Wikipedia Knee interactive chart from {raw_csv}...")
        df = pd.read_csv(raw_csv)
        knee_chart = visualizer.plot_interactive_knee(df, knee_pv=534.0, knee_rank=500)
        knee_chart.save(knee_html)
        print(f"Saved interactive chart to: {knee_html}")