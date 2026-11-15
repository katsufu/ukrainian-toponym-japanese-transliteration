import builtins
import io
import math
import re
import sys
import pandas as pd
import panphon
import panphon.distance

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Feature Edit Distance (FED) with PanPhon
class FEDEvaluator:

    def __init__(self):
        # Ensure UTF-8 file reading on Windows (cp932 locale) when PanPhon loads its IPA feature CSVs
        orig_open = builtins.open
        orig_io_open = io.open
        orig_read_csv = pd.read_csv

        def _utf8_open(file, mode='r', buffering=-1, encoding=None, *args, **kwargs):
            if 'b' not in mode and encoding is None:
                encoding = 'utf-8'
            return orig_open(file, mode, buffering, encoding, *args, **kwargs)

        def _utf8_read_csv(filepath_or_buffer, *args, **kwargs):
            if hasattr(filepath_or_buffer, 'reconfigure'):
                filepath_or_buffer.reconfigure(encoding='utf-8')
            if 'encoding' not in kwargs or kwargs['encoding'] is None:
                kwargs['encoding'] = 'utf-8'
            return orig_read_csv(filepath_or_buffer, *args, **kwargs)

        builtins.open = _utf8_open
        io.open = _utf8_open
        pd.read_csv = _utf8_read_csv
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.startswith('panphon') and hasattr(mod, 'open'):
                setattr(mod, 'open', _utf8_open)
        try:
            self.distance = panphon.distance.Distance()
        finally:
            builtins.open = orig_open
            io.open = orig_io_open
            pd.read_csv = orig_read_csv

    @staticmethod
    def _is_empty(val) -> bool:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return True
        return not str(val).strip()

    @staticmethod
    def _normalize_gt_ipa(gt_ipa: str) -> str:
        # Ensure deprecated pre-1989 affricate ligatures and ASCII stress marks ('V -> Vː)
        # are normalized even if FEDEvaluator is called directly with raw Ukrainian IPA
        ipa = (
            str(gt_ipa).strip()
            .replace('ʦ', 't͡s')
            .replace('ʣ', 'd͡z')
            .replace('ʧ', 't͡ʃ')
            .replace('ʤ', 'd͡ʒ')
        )
        ipa = re.sub(r"'([ɑɛiɪɔuɐo])", r"\1ː", ipa)
        return ipa.replace(' ', '')

    # Calculating FED between ground truth and candidates
    def evaluate(self, candidate_ipa, ground_truth_ipa):
        if self._is_empty(candidate_ipa) or self._is_empty(ground_truth_ipa):
            return float('nan')
        c_ipa = str(candidate_ipa).replace(' ', '')
        gt_ipa = self._normalize_gt_ipa(ground_truth_ipa)
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
