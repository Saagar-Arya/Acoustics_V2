from __future__ import annotations

from typing import List, Optional
import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft, ifft, fftfreq

import Hydrophone
import struct
import glob


class HydrophoneArray:
    """Manages an array of hydrophones with signal processing and time-of-arrival detection capabilities."""

    def __init__(
        self,
        sampling_freq: float = 781250,
        search_band_min: float = 25000,
        search_band_max: float = 40000,
        bandwidth: float = 100.0,
        data_collection: bool = False,
        data_collection_path: str = "",
        data_collection_target: int = 0
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

        self.data_collection = data_collection
        self.data_collection_path = data_collection_path
        self.data_collection_target = data_collection_target
        self.headers = [
            "Target_Hydrophone",
            "Earliest_Hydrophone_TOA",
            "Hydrophone_0_TOA",
            "Hydrophone_1_TOA",
            "Hydrophone_2_TOA",
            "Hydrophone_3_TOA",
            "Earliest_Hydrophone_GCC",
            "Hydrophone_0_TDOA",
            "Hydrophone_1_TDOA",
            "Hydrophone_2_TDOA",
            "Hydrophone_3_TDOA"
        ]

        if self.data_collection:
            self._setup_data_collection()

    # Goal: Load time-voltage data from a CSV file into hydrophone array
    # How: Detects and skips header rows, then populates each hydrophone with time and voltage data
    # Return: None (modifies hydrophone objects in place)
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
    # How: Detects and skips header rows, then populates each hydrophone with time and voltage data
    # Return: None (modifies hydrophone objects in place)
    def load_from_bin(self, folder: str) -> None:
        self.reset_selected()

        # Sampling frequency fallback if not set on the instance
        fs = getattr(self, "sampling_freq", 1_250_000.0)

        # Collect all per-channel binary files (e.g., TEMP_A0.bin, TEMP_A1.bin, …)
        files = sorted(glob.glob(os.path.join(folder, "TEMP_A*.bin")))
        if not files:
            raise FileNotFoundError(f"No TEMP_A*.bin files in {folder}")

        # Inner reader: parse one Saleae binary file and return (begin_time, sample_rate, samples)
        def _read_one(p, _fs):
            with open(p, "rb") as f:
                h = f.read(8)
                if h == b"<SALEAE>":
                    # New-format header with metadata (different types of saleae logic binary files exist)
                    struct.unpack("<I", f.read(4))      # version
                    struct.unpack("<I", f.read(4))      # type
                    bt = struct.unpack("<d", f.read(8))[0]
                    sr = struct.unpack("<d", f.read(8))[0]
                    struct.unpack("<d", f.read(8))      # downsample
                    n = struct.unpack("<I", f.read(4))[0]
                    x = np.fromfile(f, dtype="<f4", count=n)
                    return bt, sr, x
                
                # Legacy/simple variant: first 8 bytes are sample count (uint64), then 8 bytes reserved
                n = struct.unpack("<Q", h)[0]
                f.read(8)                               # skip two uint32
                x = np.fromfile(f, dtype="<f4", count=n)
                return 0.0, _fs, x

        # Read the first channel to establish timeline and reference length
        bt, fs0, x0 = _read_one(files[0], fs)
        n = len(x0)
        # Construct absolute time vector using file begin time and sample rate
        t = bt + np.arange(n, dtype=np.float64) / fs0
        self.hydrophones[0].times = t
        self.hydrophones[0].voltages = x0

        # Read remaining channels; align by trimming to the first channel's length
        for i, p in enumerate(files[1:], start=1):
            _, _, xi = _read_one(p, fs0)
            xi = xi[:n]
            self.hydrophones[i].times = t[:len(xi)]
            self.hydrophones[i].voltages = xi
        # If there are more hydrophone objects than files, clear the extras
        for j in range(len(files), len(self.hydrophones)):
            self.hydrophones[j].reset()

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

    # Goal: Initialize CSV file for data collection
    # How: Creates directory and writes header row to CSV file
    # Return: None (creates file on disk)
    def _setup_data_collection(self) -> None:
        os.makedirs(os.path.dirname(self.data_collection_path), exist_ok=True)

        with open(self.data_collection_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)

        print(f"Data Collection CSV file created at {self.data_collection_path} with headers: {self.headers}")

    # Goal: Append TOA detection results to data collection CSV
    # How: Identifies earliest hydrophone and writes target ID, earliest index, and all TOAs to CSV
    # Return: None (appends row to CSV file)
    def append_envelope_data(self) -> None:
        row_data = [self.data_collection_target]

        sorted_toa = sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (not ih[1].found_peak, ih[1].toa_time if ih[1].toa_time is not None else float('inf'))
        )

        earliest_hydrophone_idx = sorted_toa[0][0]
        row_data.append(earliest_hydrophone_idx)
        print(f"Earliest hydrophone (envelope): {earliest_hydrophone_idx}")

        row_data.extend([h.toa_time for h in self.hydrophones])

        with open(self.data_collection_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row_data)

    # Goal: Append combined TOA and GCC-PHAT results to data collection CSV
    # How: Identifies earliest hydrophone for both methods and writes all timing data to CSV
    # Return: None (appends row to CSV file)
    def append_combined_data(self) -> None:
        row_data = [self.data_collection_target]

        # Find earliest hydrophone by envelope TOA
        sorted_toa = sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (not ih[1].found_peak, ih[1].toa_time if ih[1].toa_time is not None else float('inf'))
        )
        earliest_toa_idx = sorted_toa[0][0]
        row_data.append(earliest_toa_idx)
        print(f"Earliest hydrophone (envelope): {earliest_toa_idx}")

        # Append each hydrophone's TOA
        row_data.extend([h.toa_time for h in self.hydrophones])

        # Find earliest hydrophone by GCC-PHAT TDOA (smallest TDOA relative to reference)
        sorted_gcc = sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (ih[1].gcc_tdoa is None, ih[1].gcc_tdoa if ih[1].gcc_tdoa is not None else float('inf'))
        )
        earliest_gcc_idx = sorted_gcc[0][0]
        row_data.append(earliest_gcc_idx)
        print(f"Earliest hydrophone (GCC): {earliest_gcc_idx}")

        # Append each hydrophone's TDOA
        row_data.extend([h.gcc_tdoa for h in self.hydrophones])

        with open(self.data_collection_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row_data)



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
