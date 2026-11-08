import math
import re
try:
    import epitran
except ImportError:
    epitran = None

# Japanese G2P, supporting standard and extended Katakana digraphs for Ukrainian transliteration
class JapaneseG2P:

    # 1. Extended & standard 2-character Katakana digraphs (morae) -> (onset_consonant, nucleus_vowel_or_glide)
    #    Distinguishes palatalized I-row (リ=ɾʲi, ミ=mʲi, ニ=ɲi, キ=kʲi, ピ=pʲi, ビ=bʲi, シ=ɕi, ジ=ʑi, チ=t͡ɕi)
    #    from non-palatalized extended digraphs (ルィ=ɾi, ムィ=mi, ヌィ=ni, クィ=ki, プィ=pi, ブィ=bi, スィ=si, ズィ=zi, ティ/トィ=ti, ディ/ドィ=di)
    DIGRAPH_MAP = {
        # Vocalic / glide digraphs
        'イィ': ('j', 'i'),
        'イェ': ('j', 'e'),
        'ウィ': ('w', 'i'),
        'ウェ': ('w', 'e'),
        'ウォ': ('w', 'o'),
        'ヴァ': ('v', 'a'),
        'ヴィ': ('vʲ', 'i'),
        'ヴェ': ('v', 'e'),
        'ヴォ': ('v', 'o'),
        'ヴャ': ('vʲ', 'a'),
        'ヴュ': ('vʲ', 'ɯ'),
        'ヴョ': ('vʲ', 'o'),
        # Non-palatalized C + ィ / ゥ / ァ / ェ / ォ
        'ルィ': ('ɾ', 'i'),
        'ムィ': ('m', 'i'),
        'ヌィ': ('n', 'i'),
        'クィ': ('k', 'i'),
        'グィ': ('ɡ', 'i'),
        'プィ': ('p', 'i'),
        'ブィ': ('b', 'i'),
        'トィ': ('t', 'i'),
        'ドィ': ('d', 'i'),
        'ティ': ('t', 'i'),
        'ディ': ('d', 'i'),
        'トゥ': ('t', 'ɯ'),
        'ドゥ': ('d', 'ɯ'),
        'スィ': ('s', 'i'),
        'ズィ': ('z', 'i'),
        'ツァ': ('t͡s', 'a'),
        'ツィ': ('t͡s', 'i'),
        'ツェ': ('t͡s', 'e'),
        'ツォ': ('t͡s', 'o'),
        'クァ': ('k', 'wa'),
        'クェ': ('k', 'we'),
        'クォ': ('k', 'wo'),
        'グァ': ('ɡ', 'wa'),
        'ファ': ('ɸ', 'a'),
        'フィ': ('ɸ', 'i'),
        'フェ': ('ɸ', 'e'),
        'フォ': ('ɸ', 'o'),
        'フュ': ('ɸʲ', 'ɯ'),
        # Palatalized C + ャ / ュ / ョ / ェ / ィ
        'キャ': ('kʲ', 'a'),
        'キュ': ('kʲ', 'ɯ'),
        'キョ': ('kʲ', 'o'),
        'ギャ': ('ɡʲ', 'a'),
        'ギュ': ('ɡʲ', 'ɯ'),
        'ギョ': ('ɡʲ', 'o'),
        'シャ': ('ɕ', 'a'),
        'シュ': ('ɕ', 'ɯ'),
        'シェ': ('ɕ', 'e'),
        'ショ': ('ɕ', 'o'),
        'ジャ': ('ʑ', 'a'),
        'ジュ': ('ʑ', 'ɯ'),
        'ジェ': ('ʑ', 'e'),
        'ジョ': ('ʑ', 'o'),
        'ジィ': ('ʑ', 'i'),
        'チャ': ('t͡ɕ', 'a'),
        'チュ': ('t͡ɕ', 'ɯ'),
        'チェ': ('t͡ɕ', 'e'),
        'チョ': ('t͡ɕ', 'o'),
        'チィ': ('t͡ɕ', 'i'),
        'ニャ': ('ɲ', 'a'),
        'ニュ': ('ɲ', 'ɯ'),
        'ニェ': ('ɲ', 'e'),
        'ニョ': ('ɲ', 'o'),
        'ニィ': ('ɲ', 'i'),
        'ヒャ': ('ç', 'a'),
        'ヒュ': ('ç', 'ɯ'),
        'ヒョ': ('ç', 'o'),
        'ビャ': ('bʲ', 'a'),
        'ビュ': ('bʲ', 'ɯ'),
        'ビョ': ('bʲ', 'o'),
        'ピャ': ('pʲ', 'a'),
        'ピュ': ('pʲ', 'ɯ'),
        'ピョ': ('pʲ', 'o'),
        'ミャ': ('mʲ', 'a'),
        'ミュ': ('mʲ', 'ɯ'),
        'ミョ': ('mʲ', 'o'),
        'ムャ': ('mʲ', 'a'),
        'リャ': ('ɾʲ', 'a'),
        'リュ': ('ɾʲ', 'ɯ'),
        'リョ': ('ɾʲ', 'o'),
        'リィ': ('ɾʲ', 'i'),
        'テャ': ('tʲ', 'a'),
        'テュ': ('tʲ', 'ɯ'),
        'テョ': ('tʲ', 'o'),
        'デャ': ('dʲ', 'a'),
        'デュ': ('dʲ', 'ɯ'),
        'デョ': ('dʲ', 'o'),
        'ツャ': ('t͡sʲ', 'a'),
        'ツュ': ('t͡sʲ', 'ɯ'),
        'ツョ': ('t͡sʲ', 'o'),
        'スャ': ('sʲ', 'a'),
        'スュ': ('sʲ', 'ɯ'),
        'スェ': ('sʲ', 'e'),
        'ズャ': ('zʲ', 'a'),
        'ズュ': ('zʲ', 'ɯ'),
        'ンィ': ('n', 'i'),
    }

    # 2. Standard 1-character Katakana morae -> (onset_consonant, nucleus_vowel)
    MONOGRAPH_MAP = {
        'ア': ('', 'a'),   'イ': ('', 'i'),   'ウ': ('', 'ɯ'),   'エ': ('', 'e'),   'オ': ('', 'o'),
        'ァ': ('', 'a'),   'ィ': ('', 'i'),   'ゥ': ('', 'ɯ'),   'ェ': ('', 'e'),   'ォ': ('', 'o'),
        'カ': ('k', 'a'),  'キ': ('kʲ', 'i'), 'ク': ('k', 'ɯ'),  'ケ': ('k', 'e'),  'コ': ('k', 'o'),
        'ガ': ('ɡ', 'a'),  'ギ': ('ɡʲ', 'i'), 'グ': ('ɡ', 'ɯ'),  'ゲ': ('ɡ', 'e'),  'ゴ': ('ɡ', 'o'),
        'サ': ('s', 'a'),  'シ': ('ɕ', 'i'),  'ス': ('s', 'ɯ'),  'セ': ('s', 'e'),  'ソ': ('s', 'o'),
        'ザ': ('z', 'a'),  'ジ': ('ʑ', 'i'),  'ズ': ('z', 'ɯ'),  'ゼ': ('z', 'e'),  'ゾ': ('z', 'o'),
        'タ': ('t', 'a'),  'チ': ('t͡ɕ', 'i'), 'ツ': ('t͡s', 'ɯ'), 'テ': ('t', 'e'),  'ト': ('t', 'o'),
        'ダ': ('d', 'a'),  'ヂ': ('ʑ', 'i'),  'ヅ': ('z', 'ɯ'),  'デ': ('d', 'e'),  'ド': ('d', 'o'),
        'ナ': ('n', 'a'),  'ニ': ('ɲ', 'i'),  'ヌ': ('n', 'ɯ'),  'ネ': ('n', 'e'),  'ノ': ('n', 'o'),
        'ハ': ('h', 'a'),  'ヒ': ('ç', 'i'),  'フ': ('ɸ', 'ɯ'),  'ヘ': ('h', 'e'),  'ホ': ('h', 'o'),
        'バ': ('b', 'a'),  'ビ': ('bʲ', 'i'), 'ブ': ('b', 'ɯ'),  'ベ': ('b', 'e'),  'ボ': ('b', 'o'),
        'パ': ('p', 'a'),  'ピ': ('pʲ', 'i'), 'プ': ('p', 'ɯ'),  'ペ': ('p', 'e'),  'ポ': ('p', 'o'),
        'マ': ('m', 'a'),  'ミ': ('mʲ', 'i'), 'ム': ('m', 'ɯ'),  'メ': ('m', 'e'),  'モ': ('m', 'o'),
        'ヤ': ('j', 'a'),  'ユ': ('j', 'ɯ'),  'ヨ': ('j', 'o'),
        'ャ': ('j', 'a'),  'ュ': ('j', 'ɯ'),  'ョ': ('j', 'o'),
        'ラ': ('ɾ', 'a'),  'リ': ('ɾʲ', 'i'), 'ル': ('ɾ', 'ɯ'),  'レ': ('ɾ', 'e'),  'ロ': ('ɾ', 'o'),
        'ワ': ('w', 'a'),  'ヮ': ('w', 'a'),  'ヰ': ('', 'i'),   'ヱ': ('', 'e'),   'ヲ': ('', 'o'),
        'ヴ': ('v', 'ɯ'),
    }

    def __init__(self):
        self.epitran = epitran.Epitran('jpn-Kana') if epitran is not None else None

    # Converting Japanese Katakana transliteration into IPA
    def convert(self, kana_word):
        if kana_word is None or (isinstance(kana_word, float) and math.isnan(kana_word)):
            return ''
        # Strip non-phonetic delimiters (interpunct, double-hyphen, hyphen, spaces)
        kana_clean = re.sub(r'[・＝=\-－―\s]', '', str(kana_word).strip())
        if not kana_clean:
            return ''

        # Parse into mora units: digraphs first, then single kana / special morae (ー, ッ, ン)
        mora_tokens = []
        i = 0
        n = len(kana_clean)
        while i < n:
            if i + 1 < n and kana_clean[i:i+2] in self.DIGRAPH_MAP:
                mora_tokens.append(('CV', self.DIGRAPH_MAP[kana_clean[i:i+2]]))
                i += 2
            elif kana_clean[i] in self.MONOGRAPH_MAP:
                mora_tokens.append(('CV', self.MONOGRAPH_MAP[kana_clean[i]]))
                i += 1
            elif kana_clean[i] == 'ー':
                mora_tokens.append(('LONG', None))
                i += 1
            elif kana_clean[i] == 'ッ':
                mora_tokens.append(('GEM', None))
                i += 1
            elif kana_clean[i] == 'ン':
                mora_tokens.append(('NASAL', None))
                i += 1
            else:
                # Fallback for any non-standard symbol
                i += 1

        # Assemble IPA string with contextual allophony for ッ (gemination) and ン (nasal place assimilation)
        ipa_parts = []
        num_morae = len(mora_tokens)
        for idx, (m_type, m_val) in enumerate(mora_tokens):
            if m_type == 'CV':
                onset, vowel = m_val
                # Word-initial voiced sibilants in Japanese realize as affricates (d͡z / d͡ʑ)
                if idx == 0:
                    if onset == 'z':
                        onset = 'd͡z'
                    elif onset == 'ʑ':
                        onset = 'd͡ʑ'
                ipa_parts.append(onset + vowel)
            elif m_type == 'LONG':
                ipa_parts.append('ː')
            elif m_type == 'GEM':
                # Look ahead to geminate the following onset consonant
                next_onset = ''
                if idx + 1 < num_morae and mora_tokens[idx + 1][0] == 'CV':
                    next_onset = mora_tokens[idx + 1][1][0]
                if next_onset.startswith('t͡ɕ') or next_onset.startswith('t͡s'):
                    ipa_parts.append('t')
                elif next_onset.startswith('d͡ʑ') or next_onset.startswith('d͡z'):
                    ipa_parts.append('d')
                elif next_onset:
                    ipa_parts.append(next_onset)
            elif m_type == 'NASAL':
                # Place assimilation of moraic nasal ン before following consonant
                next_onset = ''
                if idx + 1 < num_morae and mora_tokens[idx + 1][0] == 'CV':
                    next_onset = mora_tokens[idx + 1][1][0]
                if next_onset.startswith(('p', 'b', 'm')):
                    ipa_parts.append('m')
                elif next_onset.startswith(('k', 'ɡ')):
                    ipa_parts.append('ŋ')
                elif next_onset.startswith(('ɲ', 'ɕ', 'ʑ', 't͡ɕ', 'd͡ʑ')):
                    ipa_parts.append('ɲ')
                elif next_onset.startswith(('t', 'd', 'n', 's', 'z', 'ɾ')):
                    ipa_parts.append('n')
                else:
                    ipa_parts.append('ɴ')

        return ''.join(ipa_parts)

if __name__ == '__main__':
    g2p = JapaneseG2P()
    test_words = ["キーウ", "リヴィウ", "ヴィーホダ", "ヴェルィーカ・ズナームヤンカ", "カームヤネツィ＝ポジーリシクィイ"]
    for w in test_words:
        print(f"{w} -> {g2p.convert(w)}")
