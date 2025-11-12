from __future__ import annotations

from typing import List, Optional
import os
import csv

import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft, ifft, fftfreq
import time

import Hydrophone

class HydrophoneArray:
    """Manages an array of hydrophones with signal processing and time-of-arrival detection capabilities."""

    def __init__(
        self,
        sampling_freq: float = 781250,
        search_band_min: float = 25000,
        search_band_max: float = 40000,
        bandwidth: float = 100.0,
        enable_data_sample: bool = False,
        data_sample_out_dir: str = "",
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.bandwidth = float(bandwidth)
        self.sampling_freq = float(sampling_freq)
        self.dt = 1 / sampling_freq

        self.hydrophones: List[Hydrophone.Hydrophone] = [
            Hydrophone.Hydrophone() for _ in range(4)
        ]

        self.threshold_factor = 0.3

        self.last_data = ""

        self.enable_data_sample = enable_data_sample 
        self.data_sample_out_dir = data_sample_out_dir
        if self.enable_data_sample:
            self.data_sample_path = self.setup_data_sample()

    def setup_data_sample(self):
        out_dir = self.data_sample_out_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        # timestamped filename to avoid clobbering existing files
        ts = time.strftime('%Y-%m-%d--%H-%M-%S')
        filename = f"data_sample_{ts}.csv"
        path = os.path.join(out_dir, filename)

        headers = [
            "Truth",
            "Envelope",
            "Envelope H0",
            "Envelope H1",
            "Envelope H2",
            "Envelope H3",
            "GCC",
            "GCC H0",
            "GCC H1",
            "GCC H2",
            "GCC H3",
        ]
        with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

        return path

    def data_sample(self, truth = None):
        if truth is None:
            if self.last_data:
                base = os.path.basename(self.last_data)
                prefix = base.split("_", 1)[0]
                if prefix.isdigit():
                    val = int(prefix)
                    if 0 <= val <= 7:
                        truth = val

        # Determine earliest hydrophone by envelope (smallest toa_time)
        envelope_candidates = [
            (i, hp.toa_time)
            for i, hp in enumerate(self.hydrophones)
            if getattr(hp, "toa_time", None) is not None and getattr(hp, "found_peak", False)
        ]
        if envelope_candidates:
            envelope_first = min(envelope_candidates, key=lambda x: x[1])[0]
        else:
            envelope_first = ""

        # Determine earliest hydrophone by GCC (smallest gcc_tdoa)
        gcc_candidates = [
            (i, hp.gcc_tdoa)
            for i, hp in enumerate(self.hydrophones)
            if getattr(hp, "gcc_tdoa", None) is not None
        ]
        if gcc_candidates:
            gcc_first = min(gcc_candidates, key=lambda x: x[1])[0]
        else:
            gcc_first = ""

        row = [
            truth,
            envelope_first,
            self.hydrophones[0].toa_time,
            self.hydrophones[1].toa_time,
            self.hydrophones[2].toa_time,
            self.hydrophones[3].toa_time,
            gcc_first,
            self.hydrophones[0].gcc_tdoa,
            self.hydrophones[1].gcc_tdoa,
            self.hydrophones[2].gcc_tdoa,
            self.hydrophones[3].gcc_tdoa,
        ]

        # Ensure data sample file exists (create header if missing)
        if not getattr(self, "data_sample_path", None):
            self.data_sample_path = self.setup_data_sample()

        # Append row to the CSV (don't overwrite header)
        with open(self.data_sample_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    # Goal: Load time-voltage data from a file into hydrophone array
    # Return: None
    def load_from_path(self, path: str)-> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bin":
            self.last_data = path
            return self.load_from_bin(path)
        elif ext == ".csv":
            self.last_data = path
            return self.load_from_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Expected .bin or .csv")
        
    # Goal: Load time-voltage data from a CSV file into hydrophone array
    # How: Detects and skips header rows, then populates each hydrophone with time and voltage data
    # Return: None
    def load_from_csv(self, path: str) -> None:
        self.reset_selected()

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
        for idx, hydrophone in enumerate(self.hydrophones):
            hydrophone.times = times
            hydrophone.voltages = data.iloc[:, idx + 1].to_numpy()

    # Goal: Load time-voltage data from a binary file into hydrophone array
    # How: Parses binary header to extract sample count, then reads voltage samples for each hydrophone channel
    # Return: None
    def load_from_bin(self, path: str) -> None:
        self.reset_selected()

        with open(path, "rb") as f:
            # Read header: 8 bytes uint64, 4 bytes uint32, 8 bytes double (little-endian)
            header = f.read(8 + 4 + 8)
            num_samples, num_channels, sample_period = struct.unpack("<QId", header)

            # read all float32 samples
            total_floats = num_samples * num_channels
            float_bytes = f.read(total_floats * 4)
            data = np.frombuffer(float_bytes, dtype="<f4")  # little-endian float32
            data = data.reshape((num_channels, num_samples))

        # Create time base
        times = np.arange(num_samples, dtype=np.float64) * sample_period
        for idx, hydrophone in enumerate(self.hydrophones):
            hydrophone.times = times
            hydrophone.voltages = data[idx]

    # Goal: Normalize selection mask to match hydrophone array length
    # How: Returns all-True mask if None, otherwise adjusts mask length to match hydrophone count
    # Return: List[bool] with length equal to number of hydrophones
    def _normalize_selection(self, selected: Optional[List[bool]]) -> List[bool]:
        if selected is None:
            return [True] * len(self.hydrophones)
        if len(selected) != len(self.hydrophones):
            return list(selected[:len(self.hydrophones)]) + [False] * max(0, len(self.hydrophones) - len(selected))
        return selected

    # Goal: Plot time-series envelope data for selected hydrophones
    # How: Creates subplots for each selected hydrophone and displays their envelope detection results
    # Return: None (displays matplotlib figure)
    def plot_selected_envelope(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        num_subplots = sum(selected)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 10), sharex=True)

        if num_subplots == 1:
            axes = [axes]

        plot_idx = 0
        for hydrophone, is_selected in zip(self.hydrophones, selected):
            if is_selected:
                self._plot_hydrophone_envelope(hydrophone, axes[plot_idx])
                plot_idx += 1

        plt.tight_layout()
        plt.show()

    # Goal: Plot envelope detection results for a single hydrophone
    # How: Displays original signal, filtered signal, envelope, and time-of-arrival marker
    # Return: None (modifies matplotlib axis in place)
    @staticmethod
    def _plot_hydrophone_envelope(hydrophone: Hydrophone.Hydrophone, ax) -> None:
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

    # Goal: Print time-of-arrival for all hydrophones sorted by detection time
    # How: Sorts hydrophones by TOA and prints each with detection status
    # Return: None (prints to console)
    def print_envelope_toas(self) -> None:
        sorted_hydrophones = sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (not ih[1].found_peak, ih[1].toa_time if ih[1].toa_time is not None else float('inf'))
        )

        for i, hydrophone in sorted_hydrophones:
            if hydrophone.toa_time is not None:
                print(f"Hydrophone {i} saw ping at {hydrophone.toa_time:.6f}s (found_peak={hydrophone.found_peak})")
            else:
                print(f"Hydrophone {i} saw ping at N/A (found_peak={hydrophone.found_peak})")

    # Goal: Apply bandpass filter to hydrophone signal using FFT
    # How: Two modes - center frequency with bandwidth, or frequency range filtering
    # Return: np.ndarray - filtered signal (also stores in hydrophone object)
    def apply_bandpass_filter(
        self,
        hydrophone: Hydrophone.Hydrophone,
        fixed_freq: Optional[float] = None,
        use_range: bool = False,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None
    ) -> np.ndarray:
        if hydrophone.voltages is None:
            raise RuntimeError("Hydrophone has no voltage data.")

        voltage_len = len(hydrophone.voltages)
        fft_vals = fft(hydrophone.voltages, n=voltage_len)
        fft_freqs = fftfreq(voltage_len, d=self.dt)

        if use_range:
            if low_freq is None:
                low_freq = max(0, self.search_band_min - 1000)
            if high_freq is None:
                high_freq = self.search_band_max + 1000

            bandpass_mask = (np.abs(fft_freqs) >= low_freq) & (np.abs(fft_freqs) <= high_freq)

            filtered_fft = np.zeros_like(fft_vals)
            filtered_fft[bandpass_mask] = fft_vals[bandpass_mask]

            filtered_signal = np.real(ifft(filtered_fft))[:voltage_len]
            hydrophone.filtered_signal_range = filtered_signal
            return filtered_signal

        else:
            if fixed_freq is None:
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
                    raise RuntimeError("No peak found in search band.")
            else:
                peak_freq = fixed_freq

            narrow_band = np.abs(fft_freqs - peak_freq) <= self.bandwidth / 2
            narrow_band |= np.abs(fft_freqs + peak_freq) <= self.bandwidth / 2

            filtered_fft = np.zeros_like(fft_vals)
            filtered_fft[narrow_band] = fft_vals[narrow_band]

            filtered_signal = np.real(ifft(filtered_fft))[:voltage_len]

            hydrophone.filtered_signal = filtered_signal
            hydrophone.peak_freq = peak_freq
            return filtered_signal
                     
    # Goal: Estimate time-of-arrival using envelope detection method
    # How: Applies bandpass filter, computes Hilbert envelope, and detects TOA via threshold crossing
    # Return: None (modifies hydrophone object with TOA data)
    def estimate_toa_by_envelope(self, hydrophone: Hydrophone.Hydrophone) -> None:
        if hydrophone.times is None or hydrophone.voltages is None:
            raise RuntimeError("Load data first with load_from_csv().")

        filtered_signal = self.apply_bandpass_filter(hydrophone)

        envelope = np.abs(hilbert(filtered_signal))

        threshold = float(self.threshold_factor) * float(np.max(envelope))
        toa_idx = np.argmax(envelope > threshold)
        toa_time = hydrophone.times[toa_idx]

        toa_idx_peak = int(np.argmax(envelope))
        toa_peak = hydrophone.times[toa_idx_peak]

        hydrophone.found_peak = True
        hydrophone.toa_idx = toa_idx
        hydrophone.toa_time = toa_time
        hydrophone.toa_peak = toa_peak
        hydrophone.envelope = envelope

    # Goal: Run envelope-based TOA estimation on selected hydrophones
    # How: Iterates through selected hydrophones and applies envelope detection to each
    # Return: None (modifies selected hydrophone objects)
    def estimate_selected_by_envelope(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        for hydrophone, is_selected in zip(self.hydrophones, selected):
            if is_selected:
                self.estimate_toa_by_envelope(hydrophone)

    # Goal: Reset selected hydrophones to clear previously loaded data
    # How: Calls reset method on each selected hydrophone object
    # Return: None (modifies hydrophone objects)
    def reset_selected(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        for hydrophone, is_selected in zip(self.hydrophones, selected):
            if is_selected:
                hydrophone.reset()

    # Goal: Compute GCC-PHAT cross-correlation between two signals for time delay estimation
    # How: Uses FFT-based cross-correlation with phase transform weighting for noise robustness
    # Return: Tuple of (cross_correlation array, lag index, time delay in seconds)
    def compute_gcc_phat(self, signal1: np.ndarray, signal2: np.ndarray) -> tuple[np.ndarray, int, float]:
        fft1 = fft(signal1)
        fft2 = fft(signal2)

        cross_spectrum = fft1 * np.conj(fft2)

        epsilon = 1e-10
        phat_weighted = cross_spectrum / (np.abs(cross_spectrum) + epsilon)

        gcc = np.real(ifft(phat_weighted))

        peak_idx = np.argmax(gcc)

        n = len(gcc)
        if peak_idx > n // 2:
            lag = peak_idx - n
        else:
            lag = peak_idx

        tdoa = lag * self.dt

        return gcc, lag, tdoa

    # Goal: Estimate time difference of arrival using GCC-PHAT for selected hydrophones
    # How: Uses first selected hydrophone as reference, applies bandpass filter, computes TDOA for others
    # Return: None (modifies hydrophone objects with TDOA data)
    def estimate_selected_by_gcc(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        selected_indices = [i for i, is_selected in enumerate(selected) if is_selected]

        if len(selected_indices) < 2:
            print("Warning: Need at least 2 hydrophones selected for GCC-PHAT TDOA estimation")
            return

        ref_idx = selected_indices[0]
        ref_hydrophone = self.hydrophones[ref_idx]

        if ref_hydrophone.voltages is None:
            raise RuntimeError("Reference hydrophone has no voltage data.")

        ref_filtered = self.apply_bandpass_filter(ref_hydrophone, use_range=True)

        for idx in selected_indices:
            hydrophone = self.hydrophones[idx]

            if hydrophone.voltages is None:
                print(f"Warning: Hydrophone {idx} has no voltage data, skipping GCC")
                continue

            if idx == ref_idx:
                hydrophone.gcc_tdoa = 0.0
                hydrophone.gcc_cc = None
                continue

            hydro_filtered = self.apply_bandpass_filter(hydrophone, use_range=True)

            gcc, lag, tdoa = self.compute_gcc_phat(ref_filtered, hydro_filtered)

            hydrophone.gcc_tdoa = tdoa
            hydrophone.gcc_cc = gcc
        

    # Goal: Print TDOA results from GCC-PHAT estimation
    # How: Iterates through selected hydrophones and displays their time delay values
    # Return: None (prints to console)
    def print_gcc_tdoa(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        for i, (hydrophone, is_selected) in enumerate(zip(self.hydrophones, selected)):
            if is_selected:
                if hydrophone.gcc_tdoa is not None:
                    print(f"Hydrophone {i}: TDOA = {hydrophone.gcc_tdoa * 1e6:.2f} μs ({hydrophone.gcc_tdoa:.9f} s)")
                else:
                    print(f"Hydrophone {i}: TDOA = N/A")

    # Goal: Plot GCC-PHAT cross-correlation results for selected hydrophones
    # How: Creates subplots showing correlation vs time delay with TDOA markers
    # Return: None (displays matplotlib figure)
    def plot_selected_gcc(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        selected_indices = [i for i, is_selected in enumerate(selected) if is_selected]

        plot_indices = [
            i for i in selected_indices
            if self.hydrophones[i].gcc_cc is not None
        ]

        if not plot_indices:
            print("No GCC-PHAT data to plot")
            return

        num_plots = len(plot_indices)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)

        if num_plots == 1:
            axes = [axes]

        for plot_idx, hydro_idx in enumerate(plot_indices):
            hydrophone = self.hydrophones[hydro_idx]
            gcc = hydrophone.gcc_cc

            n = len(gcc)
            lags = np.arange(-n // 2, n // 2 + (n % 2))
            time_lags = lags * self.dt * 1e6

            gcc_shifted = np.fft.fftshift(gcc)

            ax = axes[plot_idx]
            ax.plot(time_lags, gcc_shifted, linewidth=1)

            if hydrophone.gcc_tdoa is not None:
                tdoa_us = hydrophone.gcc_tdoa * 1e6
                ax.axvline(tdoa_us, color='r', linestyle='--',
                          label=f'TDOA = {tdoa_us:.2f} μs')

            ax.set_ylabel('Correlation')
            ax.set_title(f'Hydrophone {hydro_idx} GCC-PHAT')
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel('Time Delay (μs)')
        plt.tight_layout()
        plt.show()
