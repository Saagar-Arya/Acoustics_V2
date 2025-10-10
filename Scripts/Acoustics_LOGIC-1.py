import saleae
import time
import os 
import Take_2_bandpass_filter as HydrophoneTOA
import numpy as np
import matplotlib.pyplot as plt

class Acoustics():
    def __init__(self):
        self.LAUNCH_TIMEOUT = 15
        self.QUIET = False
        self.PORT = 10429
        self.HOST = 'localhost'
        self.LOGIC_PATH = "Logic-1.2.40-Windows/Logic-1.2.40/Logic.exe"
        self.DEVICE_SELECTION = 1    # 0 for LOGIC PRO 16, 1 for LOGIC 8, 2 for LOGIC PRO 8
        self.SAMPLING_FREQ = 781250
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
        
if __name__ == "__main__":
    # folder = time.strftime('%Y-%m-%d--%H-%M-%S')
    # path = os.path.join("Scripts",folder)
    # os.mkdir(path)

    # acoustics = Acoustics()
    # acoustics.print_saleae_status()
    # csv_path = acoustics.start_capture(1.5,path)
    # acoustics.kill_logic()
    plt.figure()

    csv_path = "Scripts/2025-10-09--18-47-46/SAMPLE.csv"
    hydrophoneTOA = HydrophoneTOA.Hydrophone_TOA()
    np_time_0,np_voltages_0 = hydrophoneTOA.csv_to_np(csv_path, 0, 1)
    np_time_1,np_voltages_1 = hydrophoneTOA.csv_to_np(csv_path, 0, 2)

    toa_0 = hydrophoneTOA.estimate_TOA(np_time_0, np_voltages_0, True)
    toa_1 = hydrophoneTOA.estimate_TOA(np_time_1, np_voltages_1, True)
   
    print(toa_0)
    print(toa_1)

    if toa_0["toa_time"] < toa_1["toa_time"]:
        print(f"First: np_time_0 ({toa_0['toa_time']:.8f}) Second: np_time_1 ({toa_1['toa_time']:.8f})")
    elif toa_0["toa_time"] > toa_1["toa_time"]:
        print(f"First: np_time_1 ({toa_1['toa_time']:.8f}) Second: np_time_0 ({toa_0['toa_time']:.8f})")
    else:
        print("Both have the same TOA time!")