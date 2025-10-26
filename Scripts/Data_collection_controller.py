import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_Array as Hydrophone_Array

SAMPLING_FREQ = 781250
SELECTED = [True, True, True, True]

prefix = ""
time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
folder = prefix + time_stamp

path = os.path.join("Scripts",folder)
os.mkdir(path)

logic = LOGIC.Logic(sampling_freq=SAMPLING_FREQ)
logic.print_saleae_status()

data_collection_path = os.path.join("Scripts",folder,"Data_Collection.csv")
hydrophone_array = Hydrophone_Array.Hydrophone_Array(sampling_freq=SAMPLING_FREQ, data_collection=True, 
                                                     data_collection_path=data_collection_path, data_collection_target = 0)

num_samples = 5
for i in range(num_samples):
    csv_path = logic.start_capture(2,path)

    hydrophone_array.csv_to_np(csv_path)

    hydrophone_array.estimate_selected_by_envelope(SELECTED)

    # hydrophone_array.plot_envelope_hydrophone(SELECTED)
    hydrophone_array.print_envelope_toas()
    print("//----------------------")

    # hydrophone_array.estimate_selected_by_gcc(SELECTED)        # compute GCC TDOAs (relative to hydrophone_0)
    # hydrophone_array.print_gcc_TDOA(SELECTED)
    # print("//----------------------")

    hydrophone_array.data_collection_envelope()

logic.kill_logic()
