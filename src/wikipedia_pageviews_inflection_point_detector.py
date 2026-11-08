import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# library for inflection point detection
from kneed import KneeLocator

## Defining the module for inflection point detection from Wikipedia views for last 2 years (Oct. 1, 2024 ~ Sep. 30, 2026)
class WikipediaViewsDetector:

    def __init__(self, curve_type='convex', direction='decreasing'):
        self.curve_type = curve_type
        self.direction = direction

    # Cleansing parentheses for disambiguation page on Wikipedia
    def clean_title(self, title):
        if pd.isna(title):
            return ''
        return re.sub(r'[\s　]*[\(（].*?[\)）]', '', str(title)).strip()

    # Calculating the knee point as threshold for Wikipedia pageviews and outputting png file for result
    def fit_and_plot(self, df, col_pv='avg_monthly_pv_2y', output_image_path='wikipedia_pageview_knee_analysis.png'):
        print('Analyzing Wikipedia Pageview distribution...')
        # Filtering missing value, make descending order for col_pv
        valid_pvs = df[col_pv].dropna().sort_values(ascending=False).values
        ranks = np.arange(1, len(valid_pvs)+1)

        # Calculating knee point, using Kneedle algorithm
        kneedle = KneeLocator(ranks, valid_pvs, curve=self.curve_type, direction=self.direction, online=True)

        if kneedle.knee is not None and kneedle.knee_y is not None:
            knee_rank = int(kneedle.knee)
            knee_pv = float(kneedle.knee_y)
        else:
            knee_rank = max(1, len(valid_pvs) // 2)
            knee_pv = float(valid_pvs[knee_rank - 1]) if len(valid_pvs) > 0 else 0.0

        print(f'Detected Knee Point: Rank = {knee_rank}, Threshold pageviews = {knee_pv:.1f} views/month')

        ## Plotting the result and saving it as an image file
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        # Plotting curve for pageviews
        ax.plot(ranks, valid_pvs, color='#1f77b4', lw=2, label='Toponyms Monthly Pageviews (2 years avg.)')

        # Highlighting of the inflection point
        ax.axvline(x=knee_rank, color='#d62728', linestyle='--', lw=1.5,
                   label=f'Knee Point (Rank: {knee_rank})')
        ax.axhline(y=knee_pv, color='#d62728', linestyle=':', lw=1.5,
                   label=f'Threshold (Pageviews: {knee_pv:.1f})')
        ax.scatter([knee_rank], [knee_pv], color='#d62728', s=80, zorder=5)

        # Styling for the graph
        ax.set_title('Wikipedia Pageviews Distribution & Knee Point Detection', fontsize=14, pad=12)
        ax.set_xlabel('Toponyms Rank (Sorted by Pageview in desc. order)', fontsize=11)
        ax.set_ylabel('Avg. Monthly Pageviews', fontsize=11)
        ax.grid(True, linestyle='--', alpha=.5)
        ax.legend(loc='upper right', fontsize=10)

        # Adding note for certain toponyms
        ax.annotate('Well-known toponyms\n(Adopt Established ones)',
                    xy=(1, valid_pvs[0]), xytext=(knee_rank*1.5, valid_pvs[0]*0.7),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1),
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='#fffae6', ec='#b38f00'))

        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()
        print(f'Plot saved successfully to: {output_image_path}')

        return knee_pv

    # Extracting established toponyms above threshold and exporting table for academic paper
    def export_established_table(self, df, knee_pv, output_csv='established_toponyms_table.csv'):
        """
        Filters toponyms where avg_monthly_pv_2y >= knee_pv and has_ja_wiki == True,
        cleans Wikipedia disambiguation titles, and exports the CSV table.
        """
        temp_df = df.copy()
        temp_df['clean_ja_wiki'] = temp_df['ja_wiki_title'].apply(self.clean_title)

        # Filter condition for established toponyms
        cond = (
            (temp_df['avg_monthly_pv_2y'] >= knee_pv) &
            (temp_df['has_ja_wiki'] == True) &
            (temp_df['clean_ja_wiki'] != '')
        )
        established_df = temp_df[cond].copy()

        # Sort descending by pageviews and deduplicate by toponym
        established_df = established_df.sort_values(by='avg_monthly_pv_2y', ascending=False)
        established_df = established_df.drop_duplicates(subset=['toponym', 'clean_ja_wiki'])

        # Format the table
        cols_to_keep = ['toponym', 'toponym_with_accent', 'clean_ja_wiki', 'avg_monthly_pv_2y']
        existing_cols = [c for c in cols_to_keep if c in established_df.columns]
        table_df = established_df[existing_cols].copy()
        table_df = table_df.rename(columns={
            'toponym': 'Ukrainian Toponym',
            'toponym_with_accent': 'Toponym (with Accent)',
            'clean_ja_wiki': 'Japanese Transliteration (Established)',
            'avg_monthly_pv_2y': '2y Avg Pageviews/Month'
        })

        # Export CSV
        table_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Exported {len(table_df)} established toponyms to: {output_csv}")

        return table_df

if __name__ == '__main__':

    # Import libraries
    import sys
    import pandas as pd

    # Processing input file
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'
    df = pd.read_csv(csv_path)

    # Instantiating detector
    detector = WikipediaViewsDetector()
    optimal_threshold = detector.fit_and_plot(
        df=df,
        output_image_path='wikipedia_pageview_knee_analysis.png'
    )
    
    # Export established toponyms table
    detector.export_established_table(df, optimal_threshold)
