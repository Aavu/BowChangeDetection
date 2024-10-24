import librosa
import numpy as np
from pitchTrack import PitchTrack
from utils import Util

def detect_bow_changes(audio_path, fs=16000, hop=160,threshold=0.03, dev="cpu", min_bow_length_ms=250) -> np.ndarray:
    """Detect bow changes in a given audio signal.
    Parameters
    ----------
    audio_path : str
        The path to the audio file.
    threshold : float optional
        The threshold for detecting a bow change.
    dev : str optional
        The device on which to run the pitch tracker.
    hop : int optional
        The hop size to use for the pitch tracker.
    fs : int optional
        The sample rate of the audio signal.
    min_bow_length_ms : int optional
        The minimum length of a bow change in milliseconds.

    Returns
    -------
    np.ndarray
        The envelope of the audio signal with bow changes masked as zeros.
    """
    pitch_tracker = PitchTrack('rmvpe', hop_size=hop, rmvpe_threshold=threshold, device=dev)

    # Load audio
    x, _ = librosa.load(audio_path, sr=fs)
    # Normalize
    x = x / np.max(np.abs(x))
    # Track pitch
    p = pitch_tracker.track(audio=x, fs=fs, return_cents=True)
    # Get valid boundaries
    si, ei = Util.get_valid_boundaries(p)
    p = p[si: ei + 1]
    x = x[si * hop: (ei + 1) * hop]

    # Get the amplitude curve and bow changes
    bins, _ = Util.pick_dips(x, fs, hop, smoothing_alpha=0.9, wait_ms=min_bow_length_ms, return_seconds=False)
    print("Bow changes at:", bins * (hop / fs))
    e = Util.envelope(x, sample_rate=fs, hop_size=hop)[:len(p)]
    e = Util.zero_lpf(e, alpha=0.9, restore_zeros=False)
    mask = np.ones_like(e)

    # Mask the bow changes
    for _i in range(1, len(bins)):
        si = bins[_i] - (hop // 128)
        ei = bins[_i]
        mask[si: ei] = 0

    return e * mask
