import os
import torch
import pyopenjtalk
import scipy.io.wavfile
from ukrainian_tts.tts import TTS, Voices, Stress

## TTS Audio Generation
# Class for generating a cache of the audio file on disk during the pre-processing stage.
class AudioGenerator:

    def __init__(self, base_audio_dir='audio'):
        self.base_dir = base_audio_dir
        os.makedirs(f'{self.base_dir}/ukrainian_ground_truth', exist_ok=True)
        os.makedirs(f'{self.base_dir}/japanese_candidates', exist_ok=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ukrainian_tts = TTS(device=device)
        self.ukrainian_voice = Voices.Dmytro.value
        self.ukrainian_stress = Stress.Model.value

    # Generating Ukrainian TTS data
    def generate_ukrainian(self, text_with_accent, uid):
        output_path = f'{self.base_dir}/ukrainian_ground_truth/{uid}.wav'
        if not os.path.exists(output_path):
            with open(output_path, 'wb') as f:
                self.ukrainian_tts.tts(text=str(text_with_accent), voice=self.ukrainian_voice,
                                       stress=self.ukrainian_stress, output_fp=f)
        return output_path

    # Generating Japanese TTS data
    def generate_japanese(self, kana_text, uid, method_name):
        output_path = f'{self.base_dir}/japanese_candidates/{uid}_{method_name}.wav'
        if not os.path.exists(output_path) and kana_text:
            x, sr = pyopenjtalk.tts(kana_text)
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
