import librosa
import crepe
import numpy as np
import torch
import pyreaper
from utils import Util
from rmvpe import rmvpe
import os


class PitchTrack:
    def __init__(self, algorithm: str,
                 hop_size = 160,
                 fmin=librosa.note_to_hz('C1'),
                 fmax=librosa.note_to_hz('C6'),
                 max_transition_rate=35.92,
                 fill_na=0, beta=(.15, 8),
                 boltzmann_parameter=.1,
                 rmvpe_threshold=0.1,
                 device='cpu'):
        self.algorithm_name = algorithm
        self.algorithm = self._pyin_track_
        self.device = device
        self.hop = hop_size

        crepe_param = {
            'hop_length': hop_size,
            'fmin': fmin,
            'fmax': fmax,
            'device': device
        }

        pyin_param = {
            'fmin': fmin,
            'fmax': fmax,
            'max_transition_rate': max_transition_rate,
            'beta_parameters': beta,
            'boltzmann_parameter': boltzmann_parameter,
            'fill_na': fill_na,
            'frame_length': 2048,
            'hop_length': hop_size
        }

        rmvpe_param = {
            'thred': rmvpe_threshold
        }

        reaper_param = {
            'minf0': fmin,
            'maxf0': fmax,
            'do_high_pass': True,
            'do_hilbert_transform': True
        }

        self.param = pyin_param
        if algorithm == 'crepe':
            self.param = crepe_param
            self.algorithm = self._crepe_track_

        if algorithm == 'rmvpe':
            dev = 'cpu' if device == 'mps' else device
            self.rmvpe = rmvpe.RMVPE(os.path.join("rmvpe", "rmvpe.pt"), is_half=False, hop_size=hop_size, device=dev)
            self.algorithm = self.rmvpe.infer_from_audio
            self.param = rmvpe_param

        if algorithm == 'reaper':
            self.param = reaper_param
            self.algorithm = self._reaper_track_

    def track(self, audio, fs, return_cents: bool = False, fill_gaps=True, zero_threshold=0.001, gap_threshold=50):
        if type(audio) == torch.Tensor:
            audio = audio.detach().cpu().numpy()

        out = self.algorithm(audio=audio, sample_rate=fs, **self.param)
        zeros = np.argwhere(out < 1).flatten()
        if return_cents:
            out[zeros] = 1
            out = librosa.hz_to_midi(out)
        out[zeros] = 0

        if fill_gaps:
            out = Util.fill_gaps(out, threshold=gap_threshold, zero_threshold=zero_threshold)

        if self.algorithm_name == 'pyin' or self.algorithm_name == 'reaper':
            out = Util.zero_lpf(out, 0.3, ignore_zeros=True)

        return out

    def _pyin_track_(self, audio, sample_rate, **kwargs):
        y = librosa.pyin(y=audio, sr=sample_rate, **kwargs)[0]
        if len(y) > 1:
            y[0] = y[1]
        return y

    def _crepe_track_(self, audio, sample_rate, **kwargs):
        return crepe.predict(audio, sample_rate, viterbi=True, verbose=0)

    def _reaper_track_(self, audio: np.ndarray, sample_rate, **kwargs):
        if audio.dtype == float:
            audio = (audio * (2**15)).astype(np.int16)

        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(audio, sample_rate, frame_period=self.hop / sample_rate, **kwargs)
        # for some reason, the last value is 0. Correct it here
        f0[-1] = f0[-2]
        return f0
