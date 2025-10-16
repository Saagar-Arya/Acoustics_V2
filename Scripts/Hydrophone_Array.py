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
        std:int=3,
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.bandwidth = float(bandwidth)
        self.sampling_freq = float(sampling_freq)
        self.dt = 1 / sampling_freq

        self.std = std

        self.hydrophone_0 = Hydrophone.Hydrophone()
        self.hydrophone_1 = Hydrophone.Hydrophone()
        self.hydrophone_2 = Hydrophone.Hydrophone()
        self.hydrophone_3 = Hydrophone.Hydrophone()
        self.hydrophones = [self.hydrophone_0,self.hydrophone_1,self.hydrophone_2,self.hydrophone_3]

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
    def plot_hydrophone(hydrophone:Hydrophone.Hydrophone, ax):
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
                
    def estimate_TOA(self, hydrophone:Hydrophone.Hydrophone):
        if (hydrophone.times is None or hydrophone.voltages is None):
            raise RuntimeError("Load data first with from_csv() or from_arrays().")
        
        # Compute FFT
        voltage_len = len(hydrophone.voltages)
        fft_vals = fft(hydrophone.voltages,n=voltage_len)
        fft_freqs = fftfreq(voltage_len, d=self.dt)
        
        # Find peak given a search band
        search_band = (fft_freqs > self.search_band_min) & (fft_freqs < self.search_band_max)
        if np.any(search_band):
            freqs_in_band = fft_freqs[search_band]
            fft_in_band = fft_vals[search_band]
            peak_freq = float(freqs_in_band[np.argmax(np.abs(fft_in_band))])
        else:
            # fallback to global positive peak
            hydrophone.found_peak = False
            return
            pos = fft_freqs > 0
            peak_freq = float(fft_freqs[pos][np.argmax(np.abs(fft_vals[pos]))])

        # Narrow Band Pass Filter
        narrow_band = np.abs(np.abs(fft_freqs) - peak_freq) <= self.bandwidth
        filtered_fft = np.zeros_like(fft_vals)
        filtered_fft[narrow_band] = fft_vals[narrow_band]

        filtered_signal = np.real(ifft(filtered_fft))[:voltage_len]
        envelope = np.abs(hilbert(filtered_signal))

        env_mean = float(np.mean(envelope))
        env_std = float(np.std(envelope))
        env_max = float(np.max(envelope))

        if env_std < 1e-12:
            min_abs_thresh = 1e-6
            if env_max < min_abs_thresh:
                hydrophone.found_peak = False
                return
        else:
            if env_max < (env_mean + self.std * env_std):
                hydrophone.found_peak = False
                return

        threshold = 0.3 * np.max(envelope)
        toa_idx = np.argmax(envelope > threshold)
        toa_time = hydrophone.times[toa_idx]
        
        hydrophone.found_peak = True
        hydrophone.toa_idx = toa_idx
        hydrophone.toa_time = toa_time
        hydrophone.peak_freq = peak_freq
        hydrophone.filtered_signal = filtered_signal
        hydrophone.envelope = envelope

    def estimate_selected_TOA(self, selected: list[bool] = [True,True,True,True]):
        for hydrophone, s in zip(self.hydrophones, selected):
            if s:
                self.estimate_TOA(hydrophone)

    def print_hydrophone_toas(self):
        sorted_hydros = sorted(enumerate(self.hydrophones),
            key=lambda x: (not x[1].found_peak, x[1].toa_time)
        )

        for i, h in sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (not ih[1].found_peak, ih[1].toa_time if ih[1].toa_time is not None else float('inf'))
        ):
            if h.toa_time is not None:
                print(f"Hydrophone {i} saw ping at {h.toa_time:.6f}s (found_peak={h.found_peak})")
            else:
                print(f"Hydrophone {i} saw ping at N/A (found_peak={h.found_peak})")