from dataclasses import dataclass
from typing import Optional
import numpy as np
from read_files import AnalogData, parse_analog_binary
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hilbert

@dataclass
class HydrophoneConfig:
    flip_gcc: bool = False

@dataclass 
class HydrophoneToA:
    toa_idx: Optional[int] = None
    toa_time: Optional[float] = None
    toa_peak: Optional[float] = None
    peak_freq: Optional[float] = None
    
    envelope: Optional[np.ndarray] = None
    found_peak: bool = False

    def reset(self):
        self.toa_idx = None
        self.toa_time = None
        self.toa_peak = None
        self.peak_freq = None
        self.filtered_signal = None
        self.envelope = None
        self.found_peak = False

@dataclass
class HydrophoneGCC_PHAT:
    tdoa_gcc: Optional[float] = None
    gcc_shift_samples: Optional[int] = None
    gcc_cc: Optional[np.ndarray] = None
    gcc_lags: Optional[np.ndarray] = None

    def set_as_reference(self):
        self.tdoa_gcc = 0.0
        self.gcc_shift_samples = 0
        self.gcc_cc = None
        self.gcc_lags = None

    def reset(self):
        self.tdoa_gcc = None
        self.gcc_shift_samples = None
        self.gcc_cc = None
        self.gcc_lags = None

class Hydrophone:
    # raw data
    times: Optional[np.ndarray] = None
    voltages: Optional[np.ndarray] = None
    sample_rate: Optional[float] = None

    filtered_signal: Optional[np.ndarray] = None

    # envelope-based ToA results
    toa: HydrophoneToA = HydrophoneToA()

    # GCC-PHAT results
    gcc_phat: HydrophoneGCC_PHAT = HydrophoneGCC_PHAT()

    # --- NEW: configuration ---
    config: HydrophoneConfig = HydrophoneConfig()

    def reset_values(self):
        self.times = None
        self.voltages = None
        self.sample_rate = None
        self.filtered_signal = None
        self.toa.reset()
        self.gcc_phat.reset()

    def update_from_binary(self, filename: str):
        self.reset_values()
        analog = parse_analog_binary(filename)
        self.voltages = analog.samples
        self.sample_rate = analog.sample_rate
        self.times = np.array(range(analog.num_samples)) / analog.sample_rate + analog.begin_time
        return self
    
    def update_from_data(self, data: AnalogData):
        self.reset_values()
        self.voltages = data.samples
        self.sample_rate = data.sample_rate
        self.times = np.array(range(data.num_samples)) / data.sample_rate + data.begin_time
        return self
    
    def apply_bandpass(self, bandwidth: float, search_band_min: float, search_band_max: float) -> None:
        if self.voltages is None:
            raise RuntimeError("Hydrophone has no voltage data.")

        if self.filtered_signal is not None:
            return  # already filtered

        voltage_len = len(self.voltages)
        fft_vals = fft(self.voltages, n=voltage_len)
        fft_freqs = fftfreq(voltage_len, d=self.dt)

        # search only positive freqs for peak
        pos_mask = fft_freqs > 0
        search_band = pos_mask & (fft_freqs >= search_band_min) & (fft_freqs <= search_band_max)

        if np.any(search_band):
            freqs_in_band = fft_freqs[search_band]
            fft_in_band = fft_vals[search_band]
            peak_idx = np.argmax(np.abs(fft_in_band))
            peak_freq = float(freqs_in_band[peak_idx])
        else:
            self.filtered_signal = None
            self.toa.peak_freq = None
            self.toa.found_peak = False
            return

        # narrow band filter
        narrow_band = np.abs(np.abs(fft_freqs) - peak_freq) <= bandwidth
        filtered_fft = np.zeros_like(fft_vals)
        filtered_fft[narrow_band] = fft_vals[narrow_band]

        filtered_signal = np.real(ifft(filtered_fft))[:voltage_len]

        self.filtered_signal = filtered_signal
        self.toa.peak_freq = peak_freq
        
    def estimate_by_envelope(self, bandwidth: float, search_band_min: float, search_band_max: float):
        if self.toa.envelope is not None:
            return  # already estimated

        self.apply_bandpass(bandwidth, search_band_min, search_band_max)
        self.toa.toa_idx = None
        self.toa.toa_time = None
        self.toa.toa_peak = None
        self.toa.envelope = None

        if (self.times is None or self.voltages is None):
            raise RuntimeError("Load data first with from_csv() or from_arrays().")
        
        envelope = np.abs(hilbert(self.filtered_signal))

        threshold = 0.3 * np.max(envelope)
        toa_idx = np.argmax(envelope > threshold)
        toa_time = self.times[toa_idx]
        
        toa_idx_peak = np.argmax(envelope > np.max(envelope))
        toa_peak = self.times[toa_idx_peak]

        self.toa.found_peak = True
        self.toa.toa_idx = toa_idx
        self.toa.toa_time = toa_time
        self.toa.toa_peak = toa_peak
        self.toa.envelope = envelope

    def gcc_phat_against(self, reference: 'Hydrophone', max_tau: float=None, regularization: float=1e-8, polarity_insensitive: bool=True):
        if self.voltages is None or reference.voltages is None:
            raise RuntimeError("Both hydrophones must have voltage data.")
        if self.filtered_signal is None or reference.filtered_signal is None:
            raise RuntimeError("Both hydrophones must have filtered signals. Call apply_bandpass() first.")

        # ensure zero-mean and apply window
        self.filtered_signal = self.filtered_signal - np.mean(self.filtered_signal)
        reference.filtered_signal = reference.filtered_signal - np.mean(reference.filtered_signal)

        winh0 = np.hanning(self.filtered_signal.size)
        winh1 = np.hanning(reference.filtered_signal.size)
        h0w = self.filtered_signal * winh0
        h1w = reference.filtered_signal * winh1

        # FFT length (>= len(x)+len(y)), use next power of two
        n = h0w.size + h1w.size
        nfft = 1 << int(np.ceil(np.log2(n)))

        # compute cross-spectrum and PHAT-normalize
        SIG = np.fft.rfft(h0w, n=nfft)
        REFSIG = np.fft.rfft(h1w, n=nfft)
        R = SIG * np.conj(REFSIG)

        denom = np.abs(R)
        denom = denom + regularization * np.max(denom)
        R_phat = R / denom

        cc_full = np.fft.irfft(R_phat, n=nfft)

        # determine max_shift in samples around center
        max_shift = int(nfft // 2)
        if max_tau is not None:
            # limit by user-provided maximum lag (in seconds)
            max_shift = min(int(self.sample_rate * float(max_tau)), max_shift)

        # center around zero lag
        cc = np.concatenate((cc_full[-max_shift:], cc_full[:max_shift + 1]))
        lags_samples = np.arange(-max_shift, max_shift + 1)
        lags_seconds = lags_samples / float(self.sample_rate)

        # find peak (optionally polarity-insensitive)
        if polarity_insensitive:
            peak_idx = np.argmax(np.abs(cc))
        else:
            peak_idx = np.argmax(cc)

        shift_samples = lags_samples[peak_idx]
        tau = shift_samples / float(self.sample_rate)
        if reference.config.flip_gcc:
            tau = tau * -1

        self.gcc_phat.tdoa_gcc = float(tau)
        self.gcc_phat.gcc_cc = cc
        self.gcc_phat.gcc_lags = lags_seconds
        self.gcc_phat.gcc_shift_samples = int(shift_samples)
