import numpy as np
from typing import Tuple
from numpy import ndarray
import torch
from scipy import signal

class Util:
    @staticmethod
    def fill_zeros(x: ndarray or torch.Tensor, eps=0.01) -> ndarray or torch.Tensor:
        """
        This function is designed for causal system.
        This means, for example if the input is [0 1 1 1 1 0 0 0 0 0],
        the output will be [1 1 1 1 1 0 0 0 0 0] and not [1 1 1 1 1 1 1 1 1 1]
        :param x: input
        :param eps: threshold
        :return: filled array
        """
        if type(x) == torch.Tensor:
            _x = x.clone()
        else:
            _x = x.copy()

        for i in range(len(_x)):
            if _x[i] < eps:
                j = i + 1
                while j < len(_x) and _x[j] < eps:
                    j += 1
                if j >= len(_x):
                    break

                start = _x[i - 1] if i > 0 else _x[j]

                if type(_x) == torch.Tensor:
                    _x[i:j] = torch.linspace(start, _x[j], j - i)
                else:
                    _x[i:j] = np.linspace(start, _x[j], j - i)
        return _x

    @staticmethod
    def fill_gaps(x, threshold=50, zero_threshold=0.001) -> ndarray:
        """
        Fill space between 2 notes in midi contour if the space is < 10 samples. Call this function for each frame
        :param x: The array to be filled
        :param threshold: (Optional) number of zeros below which they should be fixed
        :param zero_threshold: (Optional) The zero threshold to compare
        :return filled data
        """
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        _i = 1
        while _i < len(x):
            if x[_i] < zero_threshold:
                num_zeros = 1
                for _j in range(_i + 1, len(x)):
                    if x[_j] < zero_threshold:
                        num_zeros += 1
                    else:
                        if _i > 0 and np.abs(x[_i - 1] - x[_j]) > 0.01 and num_zeros < threshold:
                            x[_i:_j] = x[_i - 1]
                        _i = _j
                        break
            _i += 1

        x[np.abs(x) == np.inf] = 0
        return x

    @staticmethod
    def zero_lpf(x: np.ndarray, alpha, restore_zeros=True, ignore_zeros=False):
        """
        Applies a zero-phase low-pass filter to the signal.

        Parameters
        ----------
        x : ndarray
            Input signal
        alpha : float
            Smoothing parameter
        restore_zeros : bool, optional
            Whether to restore the original zeros in the output
        ignore_zeros : bool, optional
            Whether to ignore the zeros in the signal when computing the filter

        Returns
        -------
        y : ndarray
            Filtered signal
        """
        eps = 1e-2
        y = Util.fill_zeros(x, eps=eps) if ignore_zeros else x.copy()

        for i in range(1, len(x)):
            y[i] = (alpha * y[i - 1]) + ((1 - alpha) * y[i])

        for i in range(len(x) - 2, -1, -1):
            y[i] = (alpha * y[i + 1]) + ((1 - alpha) * y[i])

        # restore NULL values
        if restore_zeros:
            y[x < eps] = 0
        return y

    @staticmethod
    def envelope(x: ndarray or torch.Tensor, sample_rate: float, hop_size: int, normalize: bool = True):
        """
        Compute the envelope of an audio signal.

        Parameters
        ----------
        x : ndarray or torch.Tensor
            Input audio signal
        sample_rate : float
            Sample rate of the input signal
        hop_size : int
            Hop size of the envelope computation
        normalize : bool, optional
            Whether to normalize the envelope by its maximum value

        Returns
        -------
        env : ndarray or torch.Tensor
            Envelope of the input signal
        """
        eps = 1e-6
        # numpy and torch computations return different lengths! Usually len(np_env) - len(torch_env) = 1
        if type(x) == torch.Tensor:
            Zxx = torch.stft(x, n_fft=hop_size * 2, hop_length=hop_size, return_complex=True)
            env = torch.sum(torch.abs(Zxx), dim=0) + eps
            mav_env = torch.max(env)
        else:
            f, t, Zxx = signal.stft(x, sample_rate, nperseg=hop_size * 2, noverlap=hop_size)
            env = np.sum(np.abs(Zxx), axis=0) + eps
            mav_env = np.max(env)

        if normalize:
            env = env / mav_env

        return env

    @staticmethod
    def pick_dips(x: np.ndarray,
                  sample_rate: float = 16000,
                  hop_size: int = 160,
                  smoothing_alpha: float = 0.9,
                  wait_ms: int = 80, return_seconds: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function takes an audio signal and returns the timing of the dips in the signal.

        Parameters
        ----------
        x : ndarray
            Input audio signal
        sample_rate : float, optional
            Sample rate of the input signal
        hop_size : int, optional
            Hop size of the envelope computation
        smoothing_alpha : float, optional
            Smoothing parameter for the envelope computation
        wait_ms : int, optional
            Minimum time difference between two consecutive dips
        return_seconds : bool, optional
            Whether to return the time instants in seconds
        Returns
        -------
        bins : ndarray
            Time instants of the dips in the signal
        lpf_e : ndarray
            Low-pass filtered envelope of the input signal
        """
        e = Util.envelope(x, sample_rate=sample_rate, hop_size=hop_size)
        lpf_e = Util.zero_lpf(e, alpha=smoothing_alpha, restore_zeros=False)
        wait_samples = (wait_ms / 1000) / (hop_size / sample_rate)

        dips = []
        entered_valley = False
        for i in range(1, len(e) - 1):
            if e[i] < lpf_e[i]:
                diff = np.diff(e[i - 1:i + 2])
                if diff[0] < diff[1]:
                    if not entered_valley:
                        dips.append([i])
                        entered_valley = True
                    else:
                        dips[-1].append(i)
            else:
                entered_valley = False

        for i in range(len(dips)):
            d = dips[i]
            dev = np.abs(lpf_e[d] - e[d])
            idx = np.argmax(dev)
            dips[i] = dips[i][idx]

        bins = dips[:]

        # rank and filter out until all dips are at least 'wait' distance apart
        def is_success():
            return (np.diff(bins) >= wait_samples).all() or len(bins) < 2

        while not is_success():
            dev = lpf_e[bins] - e[bins]
            min_idx = np.argmin(dev)

            del bins[min_idx]

        bins = np.array(bins)
        if return_seconds:
            bins = bins * (hop_size / sample_rate)
        return bins, lpf_e

    @staticmethod
    def get_valid_boundaries(x: ndarray, eps: float = 0.01) -> Tuple[int, int]:
        """
        Find the start and end indices of valid (non-zero, non-NaN, non-Inf) data in a given array.

        Parameters
        ----------
        x : ndarray
            Input array
        eps : float, optional
            Minimum value for a valid data point

        Returns
        -------
        si : int
            Start index of valid data
        ei : int
            End index of valid data
        """
        # Find the idx of first non-zero pitch
        _si = 0
        for _i in range(len(x)):
            _p = x[_i]
            if _p < eps or np.isnan(_p) or np.isinf(_p):
                continue

            _si = _i
            break

        # Find the idx of last non-zero pitch
        _ei = len(x) - 1
        for _i in range(_ei, _si - 1, -1):
            _p = x[_i]
            if _p < eps or np.isnan(_p) or np.isinf(_p):
                continue

            _ei = _i
            break

        return _si, _ei