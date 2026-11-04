import panphon
import panphon.distance

# Feature Edit Distance (FED) with PanPhon
class FEDEvaluator:

    def __init__(self):
        self.distance = panphon.distance.Distance()

    # Calculating FED between ground truth and candidates
    def evaluate(self, candidate_ipa, ground_truth_ipa):
        if not candidate_ipa or not ground_truth_ipa:
            return float('nan')
        c_ipa = candidate_ipa.replace(' ', '')
        gt_ipa = ground_truth_ipa.replace(' ', '')
        return float(self.distance.feature_edit_distance(c_ipa, gt_ipa))

if __name__ == '__main__':
    from ukrainian_g2p import UkrainianG2P
    from japanese_g2p import JapaneseG2P

    g2p_ua = UkrainianG2P()
    g2p_ja = JapaneseG2P()
    fed = FEDEvaluator()

    test_words_ua = ["Ки́їв", "Львів", "Ви́года"]
    test_words_ja = ["キーウ", "リヴィウ", "ヴィーホダ"]
    
    for w_ua, w_ja in zip(test_words_ua, test_words_ja):
        ipa_ua = g2p_ua.convert(w_ua)
        ipa_ja = g2p_ja.convert(w_ja)
        fed_result = fed.evaluate(ipa_ja, ipa_ua)

        print('++++++')
        print(f"Ukrainian: {w_ua} -> {ipa_ua}")
        print(f"Japanese : {w_ja} -> {ipa_ja}")
        print(f"FED score: {fed_result}")
        print('\n')
