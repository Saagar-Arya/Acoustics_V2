import saleae
import time
import os 
import numpy as np
from Hydrophone import Hydrophone
from read_files import parse_analog, AnalogData
import pandas as pd

class Logic():
    def __init__(self, sampling_freq: int=781250):
        self.LAUNCH_TIMEOUT = 15
        self.QUIET = False
        self.PORT = 10429
        self.HOST = 'localhost'
        self.LOGIC_PATH = "Logic-1.2.40-Windows/Logic-1.2.40/Logic.exe"
        self.DEVICE_SELECTION = 1    # 0 for LOGIC PRO 16, 1 for LOGIC 8, 2 for LOGIC PRO 8
        self.SAMPLING_FREQ = sampling_freq
        self.H0_CHANNEL = 0
        self.H1_CHANNEL = 1
        self.H2_CHANNEL = 2
        self.H3_CHANNEL = 3
        self.CHANNELS = [self.H0_CHANNEL, self.H1_CHANNEL, self.H2_CHANNEL, self.H3_CHANNEL]
        
        self.start_logic()
        self.s = saleae.Saleae(host=self.HOST, port=self.PORT, quiet=self.QUIET)
        self.launch_configure()

    def start_logic(self): 
        if (not saleae.Saleae.is_logic_running()):
            return saleae.Saleae.launch_logic(timeout=self.LAUNCH_TIMEOUT, quiet=self.QUIET, 
                                              host=self.HOST, port=self.PORT, logic_path=self.LOGIC_PATH)
        return True

    def kill_logic(self):
        saleae.Saleae.kill_logic()

    def launch_configure(self):
        self.s.select_active_device(self.DEVICE_SELECTION)
        self.s.set_active_channels(digital=None, analog=self.CHANNELS)
        self.s.set_sample_rate_by_minimum(0,self.SAMPLING_FREQ)

    def print_saleae_status(self):
        print(f"DEBUG: IS LOGIC RUNNING: {self.s.is_logic_running()}")  
        print(f"DEBUG: CONNECTED DEVICE: {self.s.get_connected_devices()}")
        print(f"DEBUG: PERFORMANCE: {self.s.get_performance()}")  
        print(f"DEBUG: ACTIVE CHANNELS: {self.s.get_active_channels()}") 
        print(f"DEBUG: POSSIBLE SAMPLING RATES: {self.s.get_all_sample_rates()}")
        print(f"DEBUG: SAMPLING RATE: {self.s.get_sample_rate()}")
        print(f"DEBUG: POSSIBLE BANDWIDTH: {self.s.get_bandwidth(self.s.get_sample_rate())}")  
        print(f"DEBUG: ANALYZERS: {self.s.get_analyzers()}")  
        
    def start_capture(self, seconds, output_dir):
        csv_path = os.path.join(output_dir,"SAMPLE.csv")
        self.s.set_capture_seconds(seconds)
        self.s.capture_start_and_wait_until_finished()
        self.s.export_data2(file_path_on_target_machine=csv_path, format='csv')
        while(not self.s.is_processing_complete()):
            time.sleep(0.5)
        return csv_path
    
def read_binary(filename: str, prev_hydrophone: Hydrophone=None) -> Hydrophone:
    with open(filename, 'rb') as f:
        result = parse_analog(f)
    times = np.array(range(result.num_samples)) / result.sample_rate + result.begin_time
    time_series = result.samples

    hyd = Hydrophone() if prev_hydrophone is None else prev_hydrophone

    hyd.times = times
    hyd.voltages = time_series
    return hyd

def read_csv(path: str, prev_hydrophones: list[Hydrophone]=None) -> list[Hydrophone]:
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
    num_hydrophones = data.shape[1] - 1
    hydrophones = [Hydrophone() for _ in range(num_hydrophones)] if prev_hydrophones is None else prev_hydrophones
    for idx in range (0,len(hydrophones)):
        hydrophones[idx].times = times
        hydrophones[idx].voltages = data.iloc[:, idx+1].to_numpy()
    return hydrophones

