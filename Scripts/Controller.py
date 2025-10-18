import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_Array as Hydrophone_Array

# folder = time.strftime('%Y-%m-%d--%H-%M-%S')
# path = os.path.join("Scripts",folder)
# os.mkdir(path)

# logic = LOGIC.Logic()
# logic.print_saleae_status()
# csv_path = logic.start_capture(2,path)
# logic.kill_logic()

#csv_path = "Scripts/2025-10-09--18-18-31_1-0/SAMPLE.csv"
#csv_path = "Scripts/2025-10-09--18-20-27_1-0/SAMPLE.csv"
#csv_path = "Scripts/2025-10-09--18-21-04_1-0/SAMPLE.csv"
csv_path = "Scripts/2025-10-09--18-43-46_0-1/SAMPLE.csv"
#csv_path = "Scripts/2025-10-09--18-47-46_0-1/SAMPLE.csv"

hydrophone_array = Hydrophone_Array.Hydrophone_Array()
hydrophone_array.hydrophone_1.flip_gcc = True
hydrophone_array.hydrophone_2.flip_gcc = False
hydrophone_array.hydrophone_3.flip_gcc = False

hydrophone_array.csv_to_np(csv_path)

selected = [True, True, False, False]
hydrophone_array.estimate_selected_by_envelope(selected)

# hydrophone_array.plot_envelope_hydrophone(selected)
hydrophone_array.print_envelope_toas()
print("//----------------------")
hydrophone_array.estimate_selected_by_gcc(selected)        # compute GCC TDOAs (relative to hydrophone_0)
hydrophone_array.print_gcc_TDOA(selected)
print("//----------------------")

