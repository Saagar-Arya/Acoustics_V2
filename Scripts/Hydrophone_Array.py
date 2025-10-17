from scipy.signal import hilbert
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import Hydrophone

class Hydrophone_Array:
    def __init__(
        self, 
        sampling_freq=781250,
        search_band_min:float=25000,
        search_band_max:float=40000, 
        bandwidth:float=100.0,
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.bandwidth = float(bandwidth)
        self.sampling_freq = float(sampling_freq)
        self.dt = 1 / sampling_freq

        self.hydrophone_0 = Hydrophone.Hydrophone()
        self.hydrophone_1 = Hydrophone.Hydrophone()
        self.hydrophone_2 = Hydrophone.Hydrophone()
        self.hydrophone_3 = Hydrophone.Hydrophone()
        self.hydrophones = [self.hydrophone_0,self.hydrophone_1,self.hydrophone_2,self.hydrophone_3]        

        self.threshold_factor = 0.3

    def csv_to_np (self, path: str):
        skip_rows = 0
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(',')
                try:
                    float(parts[0])
                    skip_rows = i
                    break
                except ValueError:
                    continue

        data = pd.read_csv(path, skiprows=skip_rows, header=None)
        
        times = data.iloc[:, 0].to_numpy()
        for idx in range (0,len(self.hydrophones)):
            self.hydrophones[idx].times = times
            self.hydrophones[idx].voltages = data.iloc[:, idx+1].to_numpy()
        
    def plot_selected_hydrophones(self, selected: list[bool] = [True,True,True,True]):
        subplots = sum(selected)
        fig, axes = plt.subplots(subplots, 1, figsize=(10, 10), sharex=True)

        plot = 0
        for hydro, s in zip(self.hydrophones, selected):
            if s:
                self.plot_hydrophone(hydro, axes[plot])
                plot+=1
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_envelope_hydrophone(hydrophone:Hydrophone.Hydrophone, ax):
        if hydrophone.found_peak is False:
            ax.text(0.5, 0.5, "No data loaded", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("ToA Detection (No Data)")
            ax.axis("off")
            return
        
        ax.plot(hydrophone.times, hydrophone.voltages, label="Original")
        if getattr(hydrophone, "filtered_signal", None) is not None:
            ax.plot(hydrophone.times, hydrophone.filtered_signal, label="Filtered")
        if getattr(hydrophone, "envelope", None) is not None:
            ax.plot(hydrophone.times, hydrophone.envelope, label="Envelope", linestyle="--")
        if getattr(hydrophone, "toa_time", None) is not None:
            ax.axvline(hydrophone.toa_time, color="r", linestyle=":", label=f"ToA = {hydrophone.toa_time:.6f}s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage")
        ax.set_title("ToA Detection")
        ax.legend(loc="best")
        ax.grid(True)

    def print_envelope_toas(self):
        for i, h in sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (not ih[1].found_peak, ih[1].toa_time if ih[1].toa_time is not None else float('inf'))
        ):
            if h.toa_time is not None:
                print(f"Hydrophone {i} saw ping at {h.toa_time:.6f}s (found_peak={h.found_peak})")
            else:
                print(f"Hydrophone {i} saw ping at N/A (found_peak={h.found_peak})")

    def bandpass_signal(self, hydrophone: Hydrophone) -> None:
        if hydrophone.voltages is None:
            raise RuntimeError("Hydrophone has no voltage data.")

        voltage_len = len(hydrophone.voltages)
        fft_vals = fft(hydrophone.voltages, n=voltage_len)
        fft_freqs = fftfreq(voltage_len, d=self.dt)

        # search only positive freqs for peak
        pos_mask = fft_freqs > 0
        search_band = pos_mask & (fft_freqs >= self.search_band_min) & (fft_freqs <= self.search_band_max)

        if np.any(search_band):
            freqs_in_band = fft_freqs[search_band]
            fft_in_band = fft_vals[search_band]
            peak_idx = np.argmax(np.abs(fft_in_band))
            peak_freq = float(freqs_in_band[peak_idx])
        else:
            hydrophone.filtered_signal = None
            hydrophone.peak_freq = None
            hydrophone.found_peak = False
            return

        # narrow band filter
        narrow_band = np.abs(np.abs(fft_freqs) - peak_freq) <= self.bandwidth
        filtered_fft = np.zeros_like(fft_vals)
        filtered_fft[narrow_band] = fft_vals[narrow_band]

        filtered_signal = np.real(ifft(filtered_fft))[:voltage_len]

        hydrophone.filtered_signal = filtered_signal
        hydrophone.peak_freq = peak_freq
                            
    def estimate_by_envelope(self, hydrophone:Hydrophone.Hydrophone):
        self.bandpass_signal(hydrophone)
        hydrophone.toa_idx = None
        hydrophone.toa_time = None
        hydrophone.toa_peak = None
        hydrophone.envelope = None

        if (hydrophone.times is None or hydrophone.voltages is None):
            raise RuntimeError("Load data first with from_csv() or from_arrays().")
        
        envelope = np.abs(hilbert(hydrophone.filtered_signal))

        threshold = 0.3 * np.max(envelope)
        toa_idx = np.argmax(envelope > threshold)
        toa_time = hydrophone.times[toa_idx]
        
        toa_idx_peak = np.argmax(envelope > np.max(envelope))
        toa_peak = hydrophone.times[toa_idx_peak]

        hydrophone.found_peak = True
        hydrophone.toa_idx = toa_idx
        hydrophone.toa_time = toa_time
        hydrophone.toa_peak = toa_peak
        hydrophone.envelope = envelope

    def estimate_selected_by_envelope(self, selected: list[bool] = [True,True,True,True]):
        for hydrophone, s in zip(self.hydrophones, selected):
            if s:
                self.estimate_by_envelope(hydrophone)

    def gcc_phat(self, h0, h1, max_tau=None, regularization=1e-8, polarity_insensitive=True):
        self.bandpass_signal(h0)
        self.bandpass_signal(h1)

        # ensure zero-mean and apply window
        h0.filtered_signal = h0.filtered_signal - np.mean(h0.filtered_signal)
        h1.filtered_signal = h1.filtered_signal - np.mean(h1.filtered_signal)

        winh0 = np.hanning(h0.filtered_signal.size)
        winh1 = np.hanning(h1.filtered_signal.size)
        h0w = h0.filtered_signal * winh0
        h1w = h1.filtered_signal * winh1

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
            max_shift = min(int(self.sampling_freq * float(max_tau)), max_shift)

        # center around zero lag
        cc = np.concatenate((cc_full[-max_shift:], cc_full[:max_shift + 1]))
        lags_samples = np.arange(-max_shift, max_shift + 1)
        lags_seconds = lags_samples / float(self.sampling_freq)

        # find peak (optionally polarity-insensitive)
        if polarity_insensitive:
            peak_idx = np.argmax(np.abs(cc))
        else:
            peak_idx = np.argmax(cc)

        shift_samples = lags_samples[peak_idx]
        tau = shift_samples / float(self.sampling_freq)

        return tau, cc, lags_seconds, shift_samples
    
    def estimate_selected_by_gcc(self,
                                 selected: list[bool] = [True, True, True, True],
                                 max_tau=None,
                                 regularization=1e-8,
                                 polarity_insensitive=True):
        """
        Estimate relative arrival times for the selected hydrophones using GCC-PHAT.

        Approach:
          - Compute pairwise taus: tau_ij = estimated (t_i - t_j) from gcc_phat(h_i, h_j).
          - Build linear system with equations t_i - t_j = tau_ij for each measured pair.
          - Solve least-squares for t = relative times (unique up to an additive constant).
          - Normalize so earliest time == 0 and store on each hydrophone as .gcc_time
            and flag .gcc_found = True for hydrophones included.

        Notes:
          - Uses your existing gcc_phat implementation which will call bandpass_signal.
          - If only one hydrophone selected, sets its gcc_time to 0.
        """
        # collect selected indices and hydrophone objects
        sel_indices = [i for i, s in enumerate(selected) if s and i < len(self.hydrophones)]
        n = len(sel_indices)

        # reset fields
        for i in sel_indices:
            h = self.hydrophones[i]
            h.gcc_time = None
            h.gcc_found = False

        if n == 0:
            return

        if n == 1:
            h = self.hydrophones[sel_indices[0]]
            h.gcc_time = 0.0
            h.gcc_found = True
            return

        # gather pairwise measurements
        pairs = []
        taus = []
        for a_idx in range(n):
            i = sel_indices[a_idx]
            for b_idx in range(a_idx + 1, n):
                j = sel_indices[b_idx]
                try:
                    tau_ij, _, _, _ = self.gcc_phat(self.hydrophones[i],
                                                    self.hydrophones[j],
                                                    max_tau=max_tau,
                                                    regularization=regularization,
                                                    polarity_insensitive=polarity_insensitive)
                except Exception:
                    # if a pair fails, skip it
                    continue

                # gcc_phat(h_i, h_j) returns tau = t_i - t_j (positive => i earlier than j)
                pairs.append((i, j))
                taus.append(float(tau_ij))

        if len(pairs) == 0:
            # no pairwise estimates available
            return

        # Build linear system A x = b where each row encodes (t_i - t_j) = tau_ij
        m = len(pairs)
        A = np.zeros((m, n), dtype=float)   # columns correspond to sel_indices order
        b = np.zeros(m, dtype=float)

        # map hydrophone index -> column in A
        idx_to_col = {hp_idx: col for col, hp_idx in enumerate(sel_indices)}

        for row, ((i, j), tau_val) in enumerate(zip(pairs, taus)):
            A[row, idx_to_col[i]] = 1.0
            A[row, idx_to_col[j]] = -1.0
            b[row] = tau_val

        # solve least squares for relative times (unique up to additive constant)
        # t_vec solves A t_vec ≈ b
        t_vec, *_ = np.linalg.lstsq(A, b, rcond=None)

        # normalize so earliest time is zero (makes printing intuitive)
        t_vec = t_vec - np.nanmin(t_vec)

        # write back to hydrophone objects
        for col, hp_idx in enumerate(sel_indices):
            self.hydrophones[hp_idx].gcc_time = float(t_vec[col])
            self.hydrophones[hp_idx].gcc_found = True

    def print_gcc_toas(self):
        """
        Print hydrophones ordered by GCC-derived arrival times (gcc_time).
        Hydrophones without gcc_time will be shown as N/A (gcc_found False).
        """
        def key_fn(item):
            i, h = item
            return (not getattr(h, "gcc_found", False),
                    getattr(h, "gcc_time", float("inf")))

        for i, h in sorted(enumerate(self.hydrophones), key=key_fn):
            if getattr(h, "gcc_time", None) is not None:
                print(f"Hydrophone {i} saw ping at {h.gcc_time:.6f}s (gcc_found={h.gcc_found})")
            else:
                print(f"Hydrophone {i} saw ping at N/A (gcc_found={getattr(h, 'gcc_found', False)})")