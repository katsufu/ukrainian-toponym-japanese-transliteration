import os
import re
import random
import nltk
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

# Phonotactic Complexity (N-gram Perplexity)
class PerplexityEvaluator:

    def __init__(self, order=3):
        self.order = order
        self.kni = KneserNeyInterpolated(order)
        self.is_trained = False

    # Training the model on text data to build an n-gram model
    def fit(self, background_words):
        tokenized = [list(w) for w in background_words]
        train_data, vocab = padded_everygram_pipeline(self.order, tokenized)
        self.kni.fit(train_data, vocab)
        self.is_trained = True

    # Loading Katakana background corpus directly from unidic-lite
    def fit_from_unidic(self, sample_size=50000, random_seed=42):
        import unidic_lite
        sys_dic = os.path.join(unidic_lite.DICDIR, 'sys.dic')
        with open(sys_dic, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        
        # Strictly Katakana words and prolonged sound mark (no middle dots or punctuation)
        words = list(set(re.findall(r'[ァ-ヴー]{2,10}', content)))
        if sample_size and len(words) > sample_size:
            rng = random.Random(random_seed)
            words = rng.sample(words, sample_size)
        
        self.fit(words)
        return len(words)

    # Retrieve perplexity of ngrams from given toponyms 
    def evaluate(self, kana_text):
        if not self.is_trained:
            # Fallback to unidic if not trained yet
            self.fit_from_unidic(sample_size=50000)
            
        if not kana_text:
            return float('nan')
        tokens = list(re.sub(r'[・\s]', '', str(kana_text).strip()))
        test_ngrams = list(nltk.ngrams(tokens, self.order, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        return float(self.kni.perplexity(test_ngrams))

if __name__ == '__main__':
    ppl = PerplexityEvaluator()
    print("Training PerplexityEvaluator on UniDic-lite Katakana corpus (50,000 words)...")
    n_words = ppl.fit_from_unidic(sample_size=50000)
    print(f"Training completed successfully with {n_words} words!\n")

    test_words_ja = ["キーウ", "リヴィウ", "ヴィーホダ", "ザポリージャ", "オデーサ"]
    for w_ja in test_words_ja:
        ppl_result = ppl.evaluate(w_ja)
        print(f"{w_ja} -> PPL: {ppl_result:.2f}")
