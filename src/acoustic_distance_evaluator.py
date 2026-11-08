import math
import os
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, AutoFeatureExtractor
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

# Acoustic Distance, using Meta MMS (layer 7) + Utterance-level CMVN + DTW
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
            # Normalize integer PCM data to [-1.0, 1.0] float32 expected by acoustic deep learning models
            if np.issubdtype(data.dtype, np.floating):
                wf = data.astype(np.float32)
            elif np.issubdtype(data.dtype, np.signedinteger):
                # Signed PCM (e.g. 16-bit: [-2^15, 2^15-1], 32-bit: [-2^31, 2^31-1]).
                # Scale dynamically by 2^(bits - 1) (i.e. 32768.0 for 16-bit) to preserve dynamic range.
                scale = float(2 ** (np.iinfo(data.dtype).bits - 1))
                wf = data.astype(np.float32) / scale
            elif np.issubdtype(data.dtype, np.unsignedinteger):
                # Unsigned PCM (e.g. 8-bit: [0, 255] with silence centered at 2^(bits - 1) = 128).
                bias = float(2 ** (np.iinfo(data.dtype).bits - 1))
                wf = (data.astype(np.float32) - bias) / bias
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

    @staticmethod
    def _preprocess_waveform(waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Convert to 16 kHz mono, trim leading/trailing synthesizer silence (keeping a 20ms margin),
        and normalize peak amplitude to 1.0.
        """
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        wf = waveform.squeeze(0)
        max_amp = torch.max(torch.abs(wf))
        if max_amp > 1e-8:
            wf = wf / max_amp
            # Trim leading/trailing silence (< 1% peak amplitude) with 20ms (320 samples) margin
            active = torch.nonzero(torch.abs(wf) > 0.01, as_tuple=False).squeeze(-1)
            if active.numel() > 0:
                start_idx = max(0, int(active[0].item()) - 320)
                end_idx = min( int(wf.shape[0]), int(active[-1].item()) + 320 )
                if end_idx - start_idx >= 800:  # Ensure at least 50ms of audio for MMS CNN kernel
                    wf = wf[start_idx:end_idx]
        return wf

    @staticmethod
    def _cmvn(feat: np.ndarray) -> np.ndarray:
        """
        Utterance-level Cepstral / Feature Mean and Variance Normalization (CMVN) along the time axis (axis=0).
        Removes time-invariant speaker/synthesizer timbre offsets while preserving phonemic trajectories.
        """
        if feat.shape[0] <= 1:
            return feat
        mean = np.mean(feat, axis=0, keepdims=True)
        std = np.std(feat, axis=0, keepdims=True)
        return (feat - mean) / (std + 1e-8)

    def __init__(self, model_name='facebook/mms-1b', target_layer=7):
        self.target_layer = target_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # Extracting features from 7th layer
    def _extract_layer7(self, audio_path):
        # Loading the WAV audio, resampling, trimming silence, and normalizing peak amplitude
        waveform, sr = self._load_audio(audio_path)
        waveform = self._preprocess_waveform(waveform, sr)

        # Passing the audio waveform through Meta MMS’s CNN feature extractor and Transformer encoder
        inputs = self.processor(waveform.numpy(), sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        # Extracting the hidden state representations from Layer 7 (T x 1280 dimensional tensor) and applying CMVN
        feat = outputs.hidden_states[self.target_layer].squeeze(0).cpu().numpy()
        return self._cmvn(feat)

    # Extracting features from 7th layer with batch
    def extract_features_batch(self, audio_paths, batch_size):
        feature_cache = {}
        valid_paths = [
            p for p in sorted(set(str(x) for x in audio_paths if x and not (isinstance(x, float) and math.isnan(x))))
            if os.path.exists(p)
        ]
        for i in range(0, len(valid_paths), batch_size):
            chunk = valid_paths[i:i + batch_size]
            waveforms = []
            for p in chunk:
                wf, sr = self._load_audio(p)
                wf = self._preprocess_waveform(wf, sr)
                waveforms.append(wf.numpy())

            raw_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
            inputs = self.processor(
                waveforms,
                sampling_rate=16000,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_values = inputs.input_values.to(self.device)
            if 'attention_mask' in inputs and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask.to(self.device)
            else:
                max_len = input_values.shape[-1]
                attention_mask = (torch.arange(max_len).unsqueeze(0) < raw_lengths.unsqueeze(1)).long().to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)
            
            # Obtain exact unpadded frame length for each sample in the batch and apply CMVN
            valid_frame_lengths = self.model._get_feat_extract_output_lengths(raw_lengths).tolist()
            layer7_all = outputs.hidden_states[self.target_layer].cpu().numpy()
            for idx, p in enumerate(chunk):
                v_len = max(1, int(valid_frame_lengths[idx]))
                feat = layer7_all[idx, :v_len]
                feature_cache[p] = self._cmvn(feat)
        return feature_cache

    def evaluate(self, ukrainian_audio_path, japanese_audio_path, feature_cache=None):
        """
        Since the speaking rate and duration differ between the Ukrainian and Japanese audio, the lengths of the frame sequences do not match.
        Cosine distance is used for vector comparison between time frames.
        Using Dynamic Time Warping, we search for an alignment path that maximizes the correspondence between the feature sets of both sequences by non-linear stretching or compressing the time axis.
        Returns the "Normalized Acoustic Distance"—the cumulative minimum cost divided by the path length (a value closer to 0 indicates greater acoustic fidelity to the original sound).
        """
        if not ukrainian_audio_path or not japanese_audio_path:
            return float('nan')
        if (isinstance(ukrainian_audio_path, float) and math.isnan(ukrainian_audio_path)) or (
            isinstance(japanese_audio_path, float) and math.isnan(japanese_audio_path)
        ):
            return float('nan')
        ukr_p = str(ukrainian_audio_path).strip()
        jpn_p = str(japanese_audio_path).strip()
        if not ukr_p or not jpn_p or not os.path.exists(ukr_p) or not os.path.exists(jpn_p):
            return float('nan')
            
        if feature_cache and ukr_p in feature_cache:
            rep_ukr = feature_cache[ukr_p]
        else:
            rep_ukr = self._extract_layer7(ukr_p)

        if feature_cache and jpn_p in feature_cache:
            rep_jpn = feature_cache[jpn_p]
        else:
            rep_jpn = self._extract_layer7(jpn_p)

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
