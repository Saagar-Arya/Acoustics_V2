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
        if h1.flip_gcc:
            tau = tau * -1
        return tau, cc, lags_seconds, shift_samples
    
    def estimate_selected_by_gcc(self, selected: list[bool] = [True, True, True, True], max_tau=None, regularization=1e-8, polarity_insensitive=True):
        """
        For each selected hydrophone (except hydrophone 0), estimate TDOA relative to hydrophone_0 using gcc_phat.
        Results are stored on each hydrophone object:
            hydrophone.tdoa_gcc (seconds)
            hydrophone.gcc_cc (cross-correlation array)
            hydrophone.gcc_lags (lags in seconds, aligned with gcc_cc)
            hydrophone.gcc_shift_samples (integer shift in samples)
        Hydrophone 0 will have tdoa_gcc = 0.0.
        """
        # ensure hydrophone_0 has data (and optionally precompute bandpass on it)
        if getattr(self.hydrophone_0, "voltages", None) is None:
            raise RuntimeError("Hydrophone 0 has no data. Load CSV first.")

        # compute bandpass for hydrophone 0 once (gcc_phat will call bandpass_signal internally too)
        # but call here explicitly to ensure peak_freq etc are set
        self.bandpass_signal(self.hydrophone_0)

        # set hydrophone 0 fields
        self.hydrophone_0.tdoa_gcc = 0.0
        self.hydrophone_0.gcc_cc = None
        self.hydrophone_0.gcc_lags = None
        self.hydrophone_0.gcc_shift_samples = 0

        for idx, (hydro, sel) in enumerate(zip(self.hydrophones, selected)):
            if not sel:
                # mark as not processed
                hydro.tdoa_gcc = None
                hydro.gcc_cc = None
                hydro.gcc_lags = None
                hydro.gcc_shift_samples = None
                continue

            if idx == 0:
                # already set above
                continue

            # ensure hydro has data
            if getattr(hydro, "voltages", None) is None:
                hydro.tdoa_gcc = None
                hydro.gcc_cc = None
                hydro.gcc_lags = None
                hydro.gcc_shift_samples = None
                continue

            # call gcc_phat with hydrophone_0 as reference and hydro as other
            try:
                tau, cc, lags_seconds, shift_samples = self.gcc_phat(self.hydrophone_0, hydro, max_tau=max_tau, regularization=regularization, polarity_insensitive=polarity_insensitive)
            except Exception as e:
                # propagate useful info on failure but keep other hydrophones processed
                hydro.tdoa_gcc = None
                hydro.gcc_cc = None
                hydro.gcc_lags = None
                hydro.gcc_shift_samples = None
                print(f"gcc_phat failed for hydrophone {idx}: {e}")
                continue

            # store results on hydrophone object
            hydro.tdoa_gcc = float(tau)
            hydro.gcc_cc = cc
            hydro.gcc_lags = lags_seconds
            hydro.gcc_shift_samples = int(shift_samples)

    def print_gcc_TDOA(self, selected: list[bool] = [True, True, True, True], indent: str = "  "):
        """
        Print TDOA results computed by estimate_selected_by_gcc for each hydrophone,
        always relative to hydrophone 0. If a hydrophone wasn't computed (tdoa_gcc is None)
        a 'N/A' is printed.
        """
        print("GCC-PHAT TDOA relative to Hydrophone 0")
        print("-" * 48)
        print(f"{'Hydrophone':<12}{'Selected':<10}{'TDOA (s)':<14}{'Shift (samples)':<16}{'Interpretation'}")
        print("-" * 48)

        for idx, (hydro, sel) in enumerate(zip(self.hydrophones, selected)):
            selected_str = "Yes" if sel else "No"
            tdoa = getattr(hydro, "tdoa_gcc", None)
            shift = getattr(hydro, "gcc_shift_samples", None)

            if tdoa is None:
                tdoa_str = "N/A"
            else:
                tdoa_str = f"{tdoa:+.6e}"  # show sign (+/-)

            if shift is None:
                shift_str = "N/A"
            else:
                shift_str = f"{shift}"

            # Interpretation: which hydrophone saw the ping first?
            if tdoa is None:
                interp = "no estimate"
            else:
                # tdoa = tau where positive means hydrophone is delayed relative to hydrophone_0 (hydrophone sees ping later)
                if idx == 0 or abs(tdoa) < 1e-12:
                    interp = "same time (reference)"
                elif tdoa > 0:
                    interp = f"Hydrophone 0 leads by {tdoa:.6e}s"
                else:
                    interp = f"Hydrophone {idx} leads by {abs(tdoa):.6e}s"

            print(f"{idx:<12}{selected_str:<10}{tdoa_str:<14}{shift_str:<16}{interp}")

        print("-" * 48)