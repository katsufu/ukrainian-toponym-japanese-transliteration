import os
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

# Acoustic Distance, using Meta MMS (layer 7) + DTW
class AcousticDistanceEvaluator:

    @staticmethod
    def _load_audio(path):
        """
        Robust audio loader supporting soundfile, scipy.io.wavfile, and torchaudio
        to bypass TorchCodec/FFmpeg backend dependencies for WAV files.
        """
        try:
            import soundfile as sf
            data, sr = sf.read(path)
            wf = torch.from_numpy(data.astype(np.float32))
            if wf.ndim == 1:
                wf = wf.unsqueeze(0)
            else:
                wf = wf.t()
            return wf, sr
        except Exception:
            pass

        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(path)
            if data.dtype == np.int16:
                wf = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                wf = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                wf = (data.astype(np.float32) - 128.0) / 128.0
            else:
                wf = data.astype(np.float32)
            wf = torch.from_numpy(wf)
            if wf.ndim == 1:
                wf = wf.unsqueeze(0)
            else:
                wf = wf.t()
            return wf, sr
        except Exception:
            pass

        return torchaudio.load(path)

    def __init__(self, model_name='facebook/mms-1b', target_layer=7):
        self.target_layer = target_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # Extracting features from 7th layer
    def _extract_layer7(self, audio_path):
        # Loading the WAV audio, resampling and normalizing it to meet the MMS model's requirements: 16 kHz, monaural, and a maximum amplitude of 1.0
        waveform, sr = self._load_audio(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # Passing the audio waveform through Meta MMS’s CNN feature extractor and Transformer encoder
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        # Extracting the hidden state representations from Layer 7 (T x 1280 dimensional tensor), where speaker-specific noise is minimized and phonemic and articulatory features are most highly concentrated
        return outputs.hidden_states[self.target_layer].squeeze(0).cpu().numpy()

    # Extracting features from 7th layer with batch
    def extract_features_batch(self, audio_paths, batch_size):
        feature_cache = {}
        valid_paths = [p for p in set(audio_paths) if os.path.exists(p)]
        for i in range(0, len(valid_paths), batch_size):
            chunk = valid_paths[i:i + batch_size]
            waveforms = []
            for p in chunk:
                wf, sr = self._load_audio(p)
                if wf.shape[0] > 1:
                    wf = torch.mean(wf, dim=0, keepdim=True)
                if sr != 16000:
                    wf = torchaudio.transforms.Resample(sr, 16000)(wf)
                wf = wf.squeeze(0)
                wf = wf / (torch.max(torch.abs(wf)) + 1e-8)
                waveforms.append(wf.numpy())
            inputs = self.processor(waveforms, sampling_rate=16000, padding=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
            
            # Obtaining and cache the length of each sample, excluding padding.
            layer7_all = outputs.hidden_states[self.target_layer].cpu().numpy()
            for idx, p in enumerate(chunk):
                feature_cache[p] = layer7_all[idx]
        return feature_cache

    def evaluate(self, ukrainian_audio_path, japanese_audio_path, feature_cache=None):
        """
        Since the speaking rate and duration differ between the Ukrainian and Japanese audio, the lengths of the frame sequences do not match.
        Cosine distance is used for vector comparison between time frames.
        Using Dynamic Time Warping, we search for an alignment path that maximizes the correspondence between the feature sets of both sequences by non-linear stretching or compressing the time axis.
        Returns the "Normalized Acoustic Distance"—the cumulative minimum cost divided by the path length (a value closer to 0 indicates greater acoustic fidelity to the original sound).
        """ 
        if not os.path.exists(ukrainian_audio_path) or not os.path.exists(japanese_audio_path):
            return float('nan')
            
        if feature_cache and ukrainian_audio_path in feature_cache:
            rep_ukr = feature_cache[ukrainian_audio_path]
        else:
            rep_ukr = self._extract_layer7(ukrainian_audio_path)

        if feature_cache and japanese_audio_path in feature_cache:
            rep_jpn = feature_cache[japanese_audio_path]
        else:
            rep_jpn = self._extract_layer7(japanese_audio_path)

        raw_distance, path = fastdtw(rep_ukr, rep_jpn, dist=cosine)
        return float(raw_distance / len(path))

if __name__ == '__main__':
    from audio_generator import AudioGenerator

    audio_gen = AudioGenerator()
    acoustic_eval = AcousticDistanceEvaluator()

    test_words_ua = ["Ки́їв", "Львів", "Ви́года"]
    test_words_ja = ["キーウ", "リヴィウ", "ヴィーホダ"]

    print("=== 1. Generating Test Audio Files ===")
    ua_audio_paths = []
    ja_audio_paths = []
    for uid, (w_ua, w_ja) in enumerate(zip(test_words_ua, test_words_ja), start=1):
        p_ua = audio_gen.generate_ukrainian(w_ua, f"mock_{uid}")
        p_ja = audio_gen.generate_japanese(w_ja, f"mock_{uid}", "mock_ja")
        ua_audio_paths.append(p_ua)
        ja_audio_paths.append(p_ja)

    print("\n=== 2. Testing Batch Feature Extraction (Meta MMS) ===")
    all_paths = ua_audio_paths + ja_audio_paths
    feature_cache = acoustic_eval.extract_features_batch(all_paths, batch_size=2)
    print(f"Cached representations for {len(feature_cache)} audio files.")

    print("\n=== 3. Testing Acoustic Distance Evaluation (Cached DTW) ===")
    for p_ua, p_ja, w_ua, w_ja in zip(ua_audio_paths, ja_audio_paths, test_words_ua, test_words_ja):
        dist = acoustic_eval.evaluate(p_ua, p_ja, feature_cache=feature_cache)
        print(f"Acoustic Distance ({w_ua} vs {w_ja}): {dist:.4f}")
