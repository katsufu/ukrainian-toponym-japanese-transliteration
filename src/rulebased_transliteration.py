import re
import pandas as pd

# Transliteration A: Using threshold of Wikipedia pageviews, retrieve rule-based transliteration or already established one
class RuleBasedTransliteration:

    def __init__(self, pv_threshold):
        self.pv_threshold = pv_threshold

    # Cleansing parentheses for disambiguation page on Wikipedia
    def clean(self, ja_toponym):
        if pd.isna(ja_toponym):
            return ''
        return re.sub(r'[\s　]*[\(（].*?[\)）]', '', str(ja_toponym)).strip()

    # Return either rule-based transliteration or already established one
    def transliterate(self, row):

        if pd.notna(row['avg_monthly_pv_2y']):
            pageviews = row['avg_monthly_pv_2y']
        else:
            pageviews = 0

        cleaned_toponym = self.clean(row.get('ja_wiki_title'))

        if pageviews >= self.pv_threshold and row.get('has_ja_wiki') == True and cleaned_toponym != '':
            return cleaned_toponym
        else:
            return str(row['japanese_transliteration_by_accent']).strip()

if __name__ == '__main__':

    # Import libraries
    import sys
    from wikipedia_pageviews_inflection_point_detector import WikipediaViewsDetector

    # Processing inputted file
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'ukrainian_toponyms_wikiinfo_with_accent_ja.csv'
    df = pd.read_csv(csv_path)
    eval_subset = df.head(10).copy()

    # Inflection point detection & plotting
    optimizer = WikipediaViewsDetector()
    optimal_threshold = optimizer.fit_and_plot(
        eval_subset, 
        col_pv='avg_monthly_pv_2y', 
        output_image_path='wikipedia_pageview_knee_analysis.png'
    )

    # Instanciating and running the function
    trial_instance = RuleBasedTransliteration(pv_threshold=optimal_threshold)

    for idx, row in eval_subset.iterrows():
        res = trial_instance.transliterate(row)
        print(f"{row['toponym']} -> {res}")