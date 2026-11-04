import re
import epitran

# Japanese G2P, using Epitran library
class JapaneseG2P:

    def __init__(self):
        self.epitran = epitran.Epitran('jpn-Kana')

    # Converting Japanese Katakana transliteration into IPA
    def convert(self, kana_word):
        kana_clean = re.sub(r'[・\s]', '', str(kana_word).strip())
        return self.epitran.transliterate(kana_clean)

if __name__ == '__main__':
    g2p = JapaneseG2P()
    test_words = ["キーウ", "リヴィウ", "ヴィーホダ"]
    for w in test_words:
        print(f"{w} -> {g2p.convert(w)}")
