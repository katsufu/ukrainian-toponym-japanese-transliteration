import re
from src.phonemizer.processor import UkrainianPhonemizer

class UkrainianG2P:

    def __init__(self):
        self._phonemizer = UkrainianPhonemizer()

    @staticmethod
    def normalize_ipa(raw_ipa: str) -> str:
        if not raw_ipa:
            return ''
        # 1. Replace deprecated pre-1989 Unicode ligatures with standard IPA tie-bar affricates
        #    recognized by PanPhon (preserving Ukrainian postalveolar t͡ʃ / d͡ʒ distinct from Japanese t͡ɕ / d͡ʑ)
        ipa = (
            str(raw_ipa).strip()
            .replace('ʦ', 't͡s')
            .replace('ʣ', 'd͡z')
            .replace('ʧ', 't͡ʃ')
            .replace('ʤ', 'd͡ʒ')
        )
        # 2. Convert lexical stress (' + vowel) into vowel + IPA length mark (ː)
        #    so PanPhon preserves Ukrainian stress (наголос) as [+long] instead of ignoring ASCII '''
        ipa = re.sub(r"'([ɑɛiɪɔuɐo])", r"\1ː", ipa)
        return ipa

    # Converting Ukrainian toponyms with accent into IPA
    def convert(self, word_with_accent):
        if word_with_accent is None:
            return ''
        raw_ipa = self._phonemizer.phonemize(str(word_with_accent).strip())
        return self.normalize_ipa(raw_ipa)

if __name__ == '__main__':
    g2p = UkrainianG2P()
    test_words = ["Ки́їв", "Львів", "Ви́года"]
    for w in test_words:
        print(f"{w} -> {g2p.convert(w)}")
