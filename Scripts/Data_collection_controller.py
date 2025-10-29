import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_Array

SAMPLING_FREQ = 781250
SELECTED = [True, True, True, True]

prefix = ""
time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
folder = prefix + time_stamp

path = os.path.join("Scripts", folder)
os.mkdir(path)

logic = LOGIC.Logic(sampling_freq=SAMPLING_FREQ)
logic.print_saleae_status()

data_collection_path = os.path.join("Scripts", folder, "Data_Collection.csv")
hydrophone_array = Hydrophone_Array.HydrophoneArray(
    sampling_freq=SAMPLING_FREQ,
    data_collection=True,
    data_collection_path=data_collection_path,
    data_collection_target=0
)

num_samples = 5
for i in range(num_samples):
    #csv_path = logic.start_capture(2, path)
    csv_path = "Scripts/2025-10-09--18-43-46_0-1/SAMPLE.csv"

    hydrophone_array.load_from_csv(csv_path)

    hydrophone_array.estimate_selected_by_envelope(SELECTED)
    hydrophone_array.print_envelope_toas()
    print("=" * 30)

    hydrophone_array.estimate_selected_by_gcc(SELECTED)
    hydrophone_array.print_gcc_tdoa(SELECTED)
    print("=" * 30)

    hydrophone_array.append_combined_data()

logic.kill_logic()

