import hashlib
import math
import os
import re
import torch
import pyopenjtalk
import scipy.io.wavfile
from ukrainian_tts.tts import TTS, Voices, Stress

## TTS Audio Generation
# Class for generating a cache of the audio file on disk during the pre-processing stage.
class AudioGenerator:

    # 1-mora carrier mapping for extended Katakana digraphs:
    # Maps digraph -> (1-mora MeCab carrier kana, modification_type, target_phoneme)
    # - 'vowel_swap': replaces the carrier mora's vowel phoneme in HTS quinphone labels (e.g., テュ ty-u -> ty-a for テャ [tʲa])
    # - 'hard_ci': keeps 1-mora duration and sets consonant right-context to +u= so HTS MLPG synthesizes
    #              a non-palatalized consonant onset [ɾ, m, n, k, ɡ, p, b] transitioning into high vowel [i/ɪ]
    # - 'direct': carrier is already in OpenJTalk's MeCab dictionary with the exact 1-mora target phoneme (e.g. ティ t-i, ディ d-i)
    EXTENDED_MORA_HTS_MAP = {
        'テャ': ('テュ', 'vowel_swap', 'a'),
        'テョ': ('テュ', 'vowel_swap', 'o'),
        'デャ': ('デュ', 'vowel_swap', 'a'),
        'デョ': ('デュ', 'vowel_swap', 'o'),
        'イィ': ('イェ', 'vowel_swap', 'i'),
        'ツャ': ('ツァ', 'direct', None),
        'ツュ': ('ツィ', 'vowel_swap', 'u'),
        'ツョ': ('ツォ', 'direct', None),
        'トィ': ('ティ', 'direct', None),
        'ドィ': ('ディ', 'direct', None),
        'スェ': ('セ',   'direct', None),
        'スャ': ('シャ', 'consonant_swap', 's'),
        'ズャ': ('ジャ', 'consonant_swap', 'z'),
        'ムャ': ('ミャ', 'direct', None),
        'ルィ': ('リ',   'hard_ci', 'u'),
        'ムィ': ('ミ',   'hard_ci', 'u'),
        'ヌィ': ('ニ',   'hard_ci', 'u'),
        'クィ': ('キ',   'hard_ci', 'u'),
        'グィ': ('ギ',   'hard_ci', 'u'),
        'プィ': ('ピ',   'hard_ci', 'u'),
        'ブィ': ('ビ',   'hard_ci', 'u'),
    }

    MORA_TOKENIZER = re.compile(r'[ァ-ヴ][ァィゥェォャュョヮ]?|ー')

    def __init__(self, base_audio_dir='audio'):
        self.base_dir = base_audio_dir
        os.makedirs(f'{self.base_dir}/ukrainian_ground_truth', exist_ok=True)
        os.makedirs(f'{self.base_dir}/japanese_candidates', exist_ok=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ukrainian_tts = TTS(device=device)
        self.ukrainian_voice = Voices.Dmytro.value
        self.ukrainian_stress = Stress.Model.value

    @staticmethod
    def _is_empty(text) -> bool:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return True
        return not str(text).strip()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(f"v2:{text}".encode('utf-8')).hexdigest()[:8]

    @classmethod
    def _synthesize_japanese_kana(cls, kana_text: str):
        # Clean non-phonetic delimiters (・, ＝, -, spaces) while preserving all phonetic kana and ー
        cleaned = re.sub(r'[・＝=\-－―\s]', '', str(kana_text).strip())
        if not cleaned:
            cleaned = str(kana_text).strip()

        morae = cls.MORA_TOKENIZER.findall(cleaned)
        if not morae:
            return pyopenjtalk.tts(cleaned)

        # Build 1-mora carrier string for MeCab frontend and record per-mora HTS label modifications
        carrier_morae = []
        mods = {}  # mora_idx -> (mod_type, target_phoneme)
        for idx, m in enumerate(morae):
            if m in cls.EXTENDED_MORA_HTS_MAP:
                carrier, mod_type, target_ph = cls.EXTENDED_MORA_HTS_MAP[m]
                carrier_morae.append(carrier)
                if mod_type != 'direct':
                    mods[idx] = (mod_type, target_ph)
            else:
                carrier_morae.append(m)

        carrier_text = ''.join(carrier_morae)
        if not mods:
            return pyopenjtalk.tts(carrier_text)

        try:
            labels = list(pyopenjtalk.extract_fullcontext(carrier_text))
            # Locate mora nuclei in HTS full-context labels:
            # Each vocalic/moraic nucleus segment has p3 in {'a','i','u','e','o','A','I','U','E','O','N','cl'}
            nucleus_indices = []
            for line_idx, line in enumerate(labels):
                m_ph = re.match(r'^([^\^]+)\^([^-]+)-([^+]+)\+([^=]+)=([^/]+)/', line)
                if m_ph:
                    p3 = m_ph.group(3)
                    if p3 in ('a', 'i', 'u', 'e', 'o', 'A', 'I', 'U', 'E', 'O', 'N', 'cl'):
                        nucleus_indices.append(line_idx)

            for mora_idx, (mod_type, target_ph) in mods.items():
                if mora_idx >= len(nucleus_indices):
                    continue
                v_idx = nucleus_indices[mora_idx]
                c_idx = v_idx - 1
                if c_idx < 0:
                    continue

                if mod_type == 'hard_ci':
                    # Set consonant right-context (+i= -> +u=) so HTS selects the non-palatalized hard consonant leaf
                    # while keeping the 1-mora high front vowel nucleus (-i+)
                    labels[c_idx] = re.sub(r'\+[iI]=', f'+{target_ph}=', labels[c_idx], count=1)
                elif mod_type == 'vowel_swap':
                    m_v = re.match(r'^([^\^]+)\^([^-]+)-([^+]+)\+([^=]+)=([^/]+)/', labels[v_idx])
                    if m_v:
                        old_v = m_v.group(3)
                        if v_idx - 2 >= 0:
                            labels[v_idx - 2] = re.sub(rf'={re.escape(old_v)}/', f'={target_ph}/', labels[v_idx - 2], count=1)
                        labels[c_idx] = re.sub(rf'\+{re.escape(old_v)}=', f'+{target_ph}=', labels[c_idx], count=1)
                        labels[v_idx] = re.sub(rf'-{re.escape(old_v)}\+', f'-{target_ph}+', labels[v_idx], count=1)
                        if v_idx + 1 < len(labels):
                            labels[v_idx + 1] = re.sub(rf'\^{re.escape(old_v)}-', f'^{target_ph}-', labels[v_idx + 1], count=1)
                        if v_idx + 2 < len(labels):
                            labels[v_idx + 2] = re.sub(rf'^{re.escape(old_v)}\^', f'{target_ph}^', labels[v_idx + 2], count=1)
                elif mod_type == 'consonant_swap':
                    m_c = re.match(r'^([^\^]+)\^([^-]+)-([^+]+)\+([^=]+)=([^/]+)/', labels[c_idx])
                    if m_c:
                        old_c = m_c.group(3)
                        if c_idx - 2 >= 0:
                            labels[c_idx - 2] = re.sub(rf'={re.escape(old_c)}/', f'={target_ph}/', labels[c_idx - 2], count=1)
                        if c_idx - 1 >= 0:
                            labels[c_idx - 1] = re.sub(rf'\+{re.escape(old_c)}=', f'+{target_ph}=', labels[c_idx - 1], count=1)
                        labels[c_idx] = re.sub(rf'-{re.escape(old_c)}\+', f'-{target_ph}+', labels[c_idx], count=1)
                        labels[v_idx] = re.sub(rf'\^{re.escape(old_c)}-', f'^{target_ph}-', labels[v_idx], count=1)
                        if v_idx + 1 < len(labels):
                            labels[v_idx + 1] = re.sub(rf'^{re.escape(old_c)}\^', f'{target_ph}^', labels[v_idx + 1], count=1)

            return pyopenjtalk.synthesize(labels)
        except Exception:
            return pyopenjtalk.tts(carrier_text)

    # Generating Ukrainian TTS data
    def generate_ukrainian(self, text_with_accent, uid):
        if self._is_empty(text_with_accent):
            return ''
        clean_text = str(text_with_accent).strip()
        thash = self._text_hash(clean_text)
        output_path = f'{self.base_dir}/ukrainian_ground_truth/{uid}_{thash}.wav'
        if not os.path.exists(output_path):
            with open(output_path, 'wb') as f:
                self.ukrainian_tts.tts(text=clean_text, voice=self.ukrainian_voice,
                                       stress=self.ukrainian_stress, output_fp=f)
        return output_path

    # Generating Japanese TTS data
    def generate_japanese(self, kana_text, uid, method_name):
        if self._is_empty(kana_text):
            return ''
        clean_kana = str(kana_text).strip()
        thash = self._text_hash(clean_kana)
        output_path = f'{self.base_dir}/japanese_candidates/{uid}_{method_name}_{thash}.wav'
        if not os.path.exists(output_path):
            x, sr = self._synthesize_japanese_kana(clean_kana)
            scipy.io.wavfile.write(output_path, sr, x.astype('int16'))
        return output_path
    
if __name__ == '__main__':

    audio_generator = AudioGenerator()

    test_words_ua = ["Ки́їв", "Львів", "Ви́года"]
    test_words_ja = ["キーウ", "リヴィウ", "ヴィーホダ"]
    
    for uid, (w_ua, w_ja) in enumerate(zip(test_words_ua, test_words_ja)):
        ground_truth_audio = audio_generator.generate_ukrainian(w_ua, str(uid))
        audio_ja = audio_generator.generate_japanese(w_ja, str(uid), 'test')
        print(f"Generated: UA -> {ground_truth_audio}, JA -> {audio_ja}")
