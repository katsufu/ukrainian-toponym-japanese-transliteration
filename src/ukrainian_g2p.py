from src.phonemizer.processor import UkrainianPhonemizer

class UkrainianG2P:

    def __init__(self):
        self._phonemizer = UkrainianPhonemizer()

    # Converting Ukrainian toponyms with accent into IPA
    def convert(self, word_with_accent):
        return self._phonemizer.phonemize(str(word_with_accent).strip())

if __name__ == '__main__':
    g2p = UkrainianG2P()
    test_words = ["Ки́їв", "Львів", "Ви́года"]
    for w in test_words:
        print(f"{w} -> {g2p.convert(w)}")
