import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_Array

SAMPLING_FREQ = 781250
SELECTED = [True, True, True, True]

# folder = time.strftime('%Y-%m-%d--%H-%M-%S')
# path = os.path.join("Scripts", folder)
# os.mkdir(path)

# logic = LOGIC.Logic(sampling_freq=SAMPLING_FREQ)
# logic.print_saleae_status()
# csv_path = logic.start_capture(0.1, path)
# # bin_folder = logic.export_binary_channel(2, path)

# logic.kill_logic()

# csv_path = "Scripts/2025-10-09--18-18-31_1-0/SAMPLE.csv"
# csv_path = "Scripts/2025-10-09--18-20-27_1-0/SAMPLE.csv"
# csv_path = "Scripts/2025-10-09--18-21-04_1-0/SAMPLE.csv"
# csv_path = "Scripts/2025-10-09--18-43-46_0-1/SAMPLE.csv"
#csv_path = "Scripts/2025-10-09--18-47-46_0-1/SAMPLE.csv"
bin_folder = "Scripts/2025-11-06--19-21-56"
# csv_path = "Scripts/2025-10-09--18-47-46_0-1/SAMPLE.csv"

hydrophone_array = Hydrophone_Array.HydrophoneArray(sampling_freq=SAMPLING_FREQ)

# hydrophone_array.load_from_csv(csv_path)
hydrophone_array.load_from_bin(bin_folder)

hydrophone_array.estimate_selected_by_envelope(SELECTED)
# hydrophone_array.plot_selected_envelope(SELECTED)
hydrophone_array.print_envelope_toas()
print("=" * 30)
hydrophone_array.estimate_selected_by_gcc(SELECTED)
# hydrophone_array.plot_selected_gcc(SELECTED)
hydrophone_array.print_gcc_tdoa(SELECTED)
print("=" * 30)


