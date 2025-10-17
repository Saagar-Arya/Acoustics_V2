from typing import Optional
import numpy as np 

class Hydrophone:
    times: Optional[np.ndarray] = None
    voltages: Optional[np.ndarray] = None

    # results
    toa_idx: Optional[int] = None
    toa_time: Optional[float] = None
    toa_peak: Optional[float] = None
    peak_freq: Optional[float] = None
    filtered_signal: Optional[np.ndarray] = None
    envelope: Optional[np.ndarray] = None
    found_peak: bool = False