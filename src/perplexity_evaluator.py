from collections import Counter, defaultdict
import math
import os
import random
import re


# Phonotactic Complexity (Mora-level N-gram Perplexity with Interpolated Modified Kneser-Ney Smoothing)
class PerplexityEvaluator:
    """
    Mora-level N-gram language model using exact Interpolated Modified Kneser-Ney (MKN) smoothing
    (Chen & Goodman, 1999, Section 2.6, Eq. 16-20 & Eq. 26) with a 0th-order uniform prior over
    the complete 841-mora Japanese + extended Katakana inventory.
    """

    # Mora tokenizer: 1 full-size Katakana + optional 1 small combining Katakana, or prolonged sound mark ー
    MORA_PATTERN = re.compile(r'[ァ-ヴ][ァィゥェォャュョヮ]?|ー')

    _SMALL_KANA = 'ァィゥェォャュョヮ'
    _BOS = '<s>'
    _EOS = '</s>'

    def __init__(self, order: int = 3):
        if order < 1:
            raise ValueError("N-gram order must be >= 1")
        self.order = order
        self.is_trained = False

        # Complete 0th-order vocabulary (841 valid Katakana morae + </s>)
        self.vocab = set(self._get_mora_inventory()) | {self._EOS}
        self.uniform_prob = 1.0 / len(self.vocab)

        # Per-order data structures: k in {1, ..., self.order}
        # counts[k][history][word] -> c_k(history, word)
        #   - For k == self.order: true empirical N-gram frequency count
        #   - For k <  self.order: unique continuation count N_{1+}(. history word)
        self.counts = {}
        self.context_totals = {}
        # context_bucket_counts[k][history] -> (N_1(h.), N_2(h.), N_{3+}(h.))
        self.context_bucket_counts = {}
        # discounts[k] -> (D_1, D_2, D_{3+})
        self.discounts = {}

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
    def _get_mora_inventory(cls) -> list:
        inventory = ['ー']
        for code in range(ord('ァ'), ord('ヴ') + 1):
            b = chr(code)
            inventory.append(b)
            for s in cls._SMALL_KANA:
                inventory.append(b + s)
        return inventory

    @staticmethod
    def _compute_mkn_discounts(ngram_counts_by_hist: dict) -> tuple:
        """
        Compute closed-form Modified Kneser-Ney discounts (D_1, D_2, D_{3+})
        from count-of-counts n_1, n_2, n_3, n_4 per Chen & Goodman (1999, Eq. 26).
        """
        coc = Counter()
        for word_counts in ngram_counts_by_hist.values():
            for c in word_counts.values():
                if 1 <= c <= 4:
                    coc[c] += 1

        n1, n2, n3, n4 = coc[1], coc[2], coc[3], coc[4]
        if n1 > 0 and n2 > 0:
            y = n1 / (n1 + 2.0 * n2)
            d1 = 1.0 - 2.0 * y * (n2 / n1)
        else:
            y = 0.5
            d1 = 0.75

        if n2 > 0 and n3 > 0:
            d2 = 2.0 - 3.0 * y * (n3 / n2)
        else:
            d2 = 1.0

        if n3 > 0 and n4 > 0:
            d3p = 3.0 - 4.0 * y * (n4 / n3)
        else:
            d3p = 1.25

        # Ensure strictly valid discount bounds 0 < D_r < r
        d1 = min(max(d1, 0.05), 0.95)
        d2 = min(max(d2, 0.10), 1.95)
        d3p = min(max(d3p, 0.15), 2.95)
        return (d1, d2, d3p)

    # Training the model on text data to build a mora-level Interpolated Modified Kneser-Ney N-gram model
    def fit(self, background_words):
        tokenized = [self.tokenize_morae(w) for w in background_words]
        tokenized = [t for t in tokenized if len(t) > 0]

        # Update vocabulary with any morae observed in training data
        self.vocab = set(self._get_mora_inventory()) | {self._EOS}
        for seq in tokenized:
            self.vocab.update(seq)
        self.uniform_prob = 1.0 / len(self.vocab)

        # 1. Extract raw empirical counts at the highest order N (and all (k+1)-grams for continuation counts)
        raw_ngrams = {k: defaultdict(Counter) for k in range(1, self.order + 1)}
        # Track unique preceding tokens v for each lower-order k-gram (history, word):
        #preceding_types[k][(history, word)] -> set of preceding token types v
        preceding_types = {k: defaultdict(set) for k in range(1, self.order)}

        left_pad = [self._BOS] * max(self.order - 1, 1)
        for seq in tokenized:
            padded = left_pad + seq + [self._EOS]
            pad_len = len(left_pad)
            # Target positions start at the first real mora (index = pad_len) through </s> (index = len(padded) - 1)
            for idx in range(pad_len, len(padded)):
                w = padded[idx]
                for k in range(1, self.order + 1):
                    hist = tuple(padded[idx - k + 1 : idx])
                    raw_ngrams[k][hist][w] += 1
                    if k < self.order:
                        prev_token = padded[idx - k]
                        preceding_types[k][(hist, w)].add(prev_token)

        # 2. Populate MKN counts:
        #    - Highest order (k == self.order): true empirical frequency counts
        #    - Lower orders  (k <  self.order): continuation counts N_{1+}(. hist w)
        self.counts = {}
        self.context_totals = {}
        self.context_bucket_counts = {}
        self.discounts = {}

        for k in range(1, self.order + 1):
            order_counts = defaultdict(dict)
            if k == self.order:
                for hist, w_counter in raw_ngrams[k].items():
                    order_counts[hist] = dict(w_counter)
            else:
                for (hist, w), prev_set in preceding_types[k].items():
                    c_cont = len(prev_set)
                    if c_cont > 0:
                        order_counts[hist][w] = c_cont

            self.counts[k] = dict(order_counts)
            self.discounts[k] = self._compute_mkn_discounts(self.counts[k])

            totals = {}
            buckets = {}
            for hist, w_dict in self.counts[k].items():
                totals[hist] = sum(w_dict.values())
                n1_h = sum(1 for c in w_dict.values() if c == 1)
                n2_h = sum(1 for c in w_dict.values() if c == 2)
                n3p_h = sum(1 for c in w_dict.values() if c >= 3)
                buckets[hist] = (n1_h, n2_h, n3p_h)

            self.context_totals[k] = totals
            self.context_bucket_counts[k] = buckets

        self.is_trained = True

    # Loading Katakana background corpus directly from unidic-lite
    def fit_from_unidic(self, sample_size: int = 50000, random_seed: int = 42) -> int:
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

    def _prob_mkn(self, word: str, history: tuple, k: int) -> float:
        """
        Recursive Interpolated Modified Kneser-Ney probability P_MKN^{(k)}(word | history).
        At k == 0, returns the exact 0th-order uniform distribution P_0(word) = 1 / |V_all|.
        """
        if k == 0:
            return self.uniform_prob

        lower_prob = self._prob_mkn(word, history[1:] if len(history) > 0 else (), k - 1)
        total_h = self.context_totals[k].get(history, 0)
        if total_h == 0:
            return lower_prob

        d1, d2, d3p = self.discounts[k]
        c_hw = self.counts[k].get(history, {}).get(word, 0)
        if c_hw == 0:
            discounted = 0.0
        elif c_hw == 1:
            discounted = max(c_hw - d1, 0.0)
        elif c_hw == 2:
            discounted = max(c_hw - d2, 0.0)
        else:
            discounted = max(c_hw - d3p, 0.0)

        n1_h, n2_h, n3p_h = self.context_bucket_counts[k][history]
        gamma_h = (d1 * n1_h + d2 * n2_h + d3p * n3p_h) / total_h

        return (discounted / total_h) + gamma_h * lower_prob

    # Retrieve perplexity of mora-level ngrams from given toponyms
    def evaluate(self, kana_text: str) -> float:
        if not self.is_trained:
            self.fit_from_unidic(sample_size=50000)

        tokens = self.tokenize_morae(kana_text)
        if not tokens:
            return float('nan')

        left_pad = [self._BOS] * max(self.order - 1, 1)
        padded = left_pad + tokens + [self._EOS]
        pad_len = len(left_pad)

        log2_prob_sum = 0.0
        num_predictions = len(tokens) + 1  # m_1, ..., m_L, </s>
        for idx in range(pad_len, len(padded)):
            w = padded[idx]
            hist = tuple(padded[idx - self.order + 1 : idx]) if self.order > 1 else ()
            p = self._prob_mkn(w, hist, self.order)
            log2_prob_sum += math.log2(max(p, 1e-15))

        cross_entropy = -log2_prob_sum / num_predictions
        return float(2.0 ** cross_entropy)


if __name__ == '__main__':
    ppl = PerplexityEvaluator(order=3)
    print("Training PerplexityEvaluator (Interpolated Modified Kneser-Ney) on UniDic-lite Katakana corpus...")
    try:
        n_words = ppl.fit_from_unidic(sample_size=50000)
        print(f"Training completed successfully with {n_words} words!")
    except ImportError:
        print("unidic_lite not installed in current env; fitting on synthetic sample corpus...")
        sample_corpus = [
            "トウキョウ", "オオサカ", "キョウト", "ヨコハマ", "ナゴヤ", "サッポロ", "フクオカ", "コウベ",
            "カワサキ", "サイタマ", "ヒロシマ", "センダイ", "チバ", "キタキュウシュウ", "サカイ", "ニイガタ",
            "ハママツ", "クマモト", "サガミハラ", "オカヤマ", "シズオカ", "カゴシマ", "ウツノミヤ", "マツヤマ",
            "キーウ", "リヴィウ", "オデーサ", "ハルキウ", "ドニプロ", "ザポリージャ", "ヴィーンヌィツャ",
        ] * 20
        ppl.fit(sample_corpus)
    for k in range(1, ppl.order + 1):
        d1, d2, d3p = ppl.discounts[k]
        print(f"  Order {k} MKN discounts: D1={d1:.4f}, D2={d2:.4f}, D3+={d3p:.4f}")

    test_words_ja = ["キーウ", "リヴィウ", "ヴィーホダ", "ザポリージャ", "オデーサ", "リュブィーミウカ", "カームヤネツィ＝ポジーリシクィイ"]
    for w_ja in test_words_ja:
        ppl_result = ppl.evaluate(w_ja)
        print(f"{w_ja} (morae={ppl.tokenize_morae(w_ja)}) -> PPL: {ppl_result:.2f}")
