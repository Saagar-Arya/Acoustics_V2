from typing import Optional
import numpy as np

class Hydrophone:
    # raw data
    times: Optional[np.ndarray] = None
    voltages: Optional[np.ndarray] = None

    # envelope-based ToA results
    toa_idx: Optional[int] = None
    toa_time: Optional[float] = None
    toa_peak: Optional[float] = None
    peak_freq: Optional[float] = None
    filtered_signal: Optional[np.ndarray] = None
    envelope: Optional[np.ndarray] = None
    found_peak: bool = False

    # GCC-PHAT results
    tdoa_gcc: Optional[float] = None
    gcc_shift_samples: Optional[int] = None
    gcc_cc: Optional[np.ndarray] = None
    gcc_lags: Optional[np.ndarray] = None

    # --- NEW: configuration ---
    flip_gcc: bool = False    # if True, invert polarity before GCC
