class Hydrophone():
    def __init__(self):
        self.times = None
        self.voltages = None

        self.toa_idx = None
        self.toa_time = None
        self.peak_freq = None
        self.filtered_signal = None
        self.envelope = None

        self.gain = 1
        self.phase = 0

        self.found_peak = False