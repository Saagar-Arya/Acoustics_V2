import Hydrophone_Array
import os
SAMPLING_FREQ = 781250
SELECTED = [True, True, True, True]

hydrophone_array = Hydrophone_Array.HydrophoneArray(sampling_freq=SAMPLING_FREQ, enable_data_sample = True, data_sample_out_dir="Scripts/Data_Collection")

base_path = "Scripts/Data_Collection"
filenames = os.listdir(base_path)
for filename in filenames:
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".bin":
        continue

    path = os.path.join(base_path,filename)
    print(path)

    hydrophone_array.load_from_path(path)

    hydrophone_array.estimate_selected_by_envelope(SELECTED)
    #hydrophone_array.plot_selected_envelope(SELECTED)
    hydrophone_array.print_envelope_toas()
    print("=" * 30)

    hydrophone_array.estimate_selected_by_gcc(SELECTED)
    #hydrophone_array.plot_selected_gcc(SELECTED)
    hydrophone_array.print_gcc_tdoa(SELECTED)
    print("=" * 30)

    hydrophone_array.data_sample()


