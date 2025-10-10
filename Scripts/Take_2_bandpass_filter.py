from scipy.signal import hilbert
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

class Hydrophone_TOA:
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
        self.sampling_freq = 1.0/sampling_freq

    @staticmethod
    def csv_to_np (path: str, time_col, votlage_col):
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

        data = pd.read_csv(path, skiprows=skip_rows, header=None) ## MIGHT NEED TO EDIT THIS DEPENDING ON SALEA OUT
        return data.iloc[:, time_col].to_numpy(), data.iloc[:,votlage_col].to_numpy()
    
    @staticmethod
    def plot_TOA(times, voltages, filtered_signal, envelope, toa_time, show=False, save_path=None):
        plt.plot(times, voltages, label="Original")
        plt.plot(times, filtered_signal, label="Filtered")
        plt.plot(times, envelope, label="Envelope", linestyle="--")
        plt.axvline(toa_time, color="r", linestyle=":", label=f"ToA = {toa_time:.6f}s")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage")
        plt.title("ToA Detection")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
        
    def estimate_TOA(self, times:np, voltages:np, show:bool = False, plot_path:str = None):
        if (times is None or voltages is None):
            raise RuntimeError("Load data first with from_csv() or from_arrays().")
        
        # Compute FFT
        voltage_len = len(voltages)
        fft_vals = fft(voltages,n=voltage_len)
        fft_freqs = fftfreq(voltage_len, d=self.sampling_freq)
        
        # Find peak given a search band
        search_band = (fft_freqs > self.search_band_min) & (fft_freqs < self.search_band_max)
        if np.any(search_band):
            freqs_in_band = fft_freqs[search_band]
            fft_in_band = fft_vals[search_band]
            peak_freq = float(freqs_in_band[np.argmax(np.abs(fft_in_band))])
        else:
            # fallback to global positive peak
            pos = fft_freqs > 0
            peak_freq = float(fft_freqs[pos][np.argmax(np.abs(fft_vals[pos]))])

        # Narrow Band Pass Filter
        narrow_band = np.abs(np.abs(fft_freqs) - peak_freq) <= self.bandwidth
        filtered_fft = np.zeros_like(fft_vals)
        filtered_fft[narrow_band] = fft_vals[narrow_band]

        filtered_signal = np.real(ifft(filtered_fft))[:voltage_len]

        envelope = np.abs(hilbert(filtered_signal))
        threshold = 0.3 * np.max(envelope)
        toa_idx = np.argmax(envelope > threshold)
        toa_time = times[toa_idx]

        if(show or plot_path):
            self.plot_TOA(times=times, voltages=voltages, filtered_signal=filtered_signal, 
                 envelope=envelope, toa_time=toa_time,show=True)
            
        return {
            "toa_idx": toa_idx,
            "toa_time": toa_time,
            "peak_freq": peak_freq     
        }