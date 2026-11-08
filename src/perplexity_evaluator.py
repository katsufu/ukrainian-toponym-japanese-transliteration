import math
import os
import re
import random
import nltk
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

# Phonotactic Complexity (Mora-level N-gram Perplexity with Kneser-Ney Smoothing)
class PerplexityEvaluator:

    # Mora tokenizer: 1 full-size Katakana + optional 1 small combining Katakana, or prolonged sound mark ー
    MORA_PATTERN = re.compile(r'[ァ-ヴ][ァィゥェォャュョヮ]?|ー')

    # Complete inventory of valid Japanese + extended Katakana morae for unigram smoothing prior
    # (Prevents NLTK KneserNeyInterpolated from returning 0.0 probability / PPL=inf on unseen extended morae
    #  per Chen & Goodman 1999 unigram uniform interpolation)
    _BASE_KANA = (
        'アイウエオカキクケコガギグゲゴサシスセソザジズゼゾタチツテトダヂヅデド'
        'ナニヌネノハヒフヘホバビブベボパピプペポマミムメモヤユヨラリルレロワヰヱヲヴンッ'
        'ァィゥェォャュョヮ'
    )
    _SMALL_KANA = 'ァィゥェォャュョヮ'

    def __init__(self, order=3):
        self.order = order
        self.kni = KneserNeyInterpolated(order)
        self.is_trained = False

    @classmethod
    def tokenize_morae(cls, text: str) -> list:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return []
        # Strip non-phonetic delimiters (・, ＝, -, spaces) while strictly preserving ー (stress/long vowel)
        cleaned = re.sub(r'[^ァ-ヴー]', '', str(text).strip())
        if not cleaned:
            return []
        return cls.MORA_PATTERN.findall(cleaned)

    @classmethod
    def _get_mora_inventory_priors(cls) -> list:
        priors = [['ー']]
        for code in range(ord('ァ'), ord('ヴ') + 1):
            b = chr(code)
            priors.append([b])
            for s in cls._SMALL_KANA:
                priors.append([b + s])
        return priors

    # Training the model on text data to build a mora-level n-gram model
    def fit(self, background_words):
        tokenized = [self.tokenize_morae(w) for w in background_words]
        tokenized = [t for t in tokenized if len(t) > 0]
        # Append 1 unigram prior per valid Katakana mora so rare extended morae (e.g., ルィ, ムィ, テャ)
        # receive a finite backoff probability instead of NLTK's 0.0 -> inf perplexity
        tokenized.extend(self._get_mora_inventory_priors())
        train_data, vocab = padded_everygram_pipeline(self.order, tokenized)
        self.kni.fit(train_data, vocab)
        self.is_trained = True

    # Loading Katakana background corpus directly from unidic-lite
    def fit_from_unidic(self, sample_size=50000, random_seed=42):
        import unidic_lite
        sys_dic = os.path.join(unidic_lite.DICDIR, 'sys.dic')
        with open(sys_dic, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        
        # Strictly Katakana words and prolonged sound mark (sorted before sampling for 100% determinism)
        words = sorted(set(re.findall(r'[ァ-ヴー]{2,10}', content)))
        if sample_size and len(words) > sample_size:
            rng = random.Random(random_seed)
            words = rng.sample(words, sample_size)
        
        self.fit(words)
        return len(words)

    # Retrieve perplexity of mora-level ngrams from given toponyms
    def evaluate(self, kana_text):
        if not self.is_trained:
            # Fallback to unidic if not trained yet
            self.fit_from_unidic(sample_size=50000)
            
        tokens = self.tokenize_morae(kana_text)
        if not tokens:
            return float('nan')
        test_ngrams = list(nltk.ngrams(tokens, self.order, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        return float(self.kni.perplexity(test_ngrams))

if __name__ == '__main__':
    ppl = PerplexityEvaluator()
    print("Training PerplexityEvaluator on UniDic-lite Katakana corpus (50,000 words)...")
    n_words = ppl.fit_from_unidic(sample_size=50000)
    print(f"Training completed successfully with {n_words} words!\n")

    test_words_ja = ["キーウ", "リヴィウ", "ヴィーホダ", "ザポリージャ", "オデーサ", "リュブィーミウカ", "カームヤネツィ＝ポジーリシクィイ"]
    for w_ja in test_words_ja:
        ppl_result = ppl.evaluate(w_ja)
        print(f"{w_ja} (morae={ppl.tokenize_morae(w_ja)}) -> PPL: {ppl_result:.2f}")
