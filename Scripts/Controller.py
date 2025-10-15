import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_Array as Hydrophone_Array

folder = time.strftime('%Y-%m-%d--%H-%M-%S')
path = os.path.join("Scripts",folder)
os.mkdir(path)

logic = LOGIC.Logic()
logic.print_saleae_status()
csv_path = logic.start_capture(1.5,path)
logic.kill_logic()

# csv_path = "Scripts/2025-10-09--18-47-46/SAMPLE.csv"
hydrophone_array = Hydrophone_Array.Hydrophone_Array()
hydrophone_array.csv_to_np(csv_path)
hydrophone_array.estimate_selected_TOA([1,1,0,0])
hydrophone_array.plot_selected_hydrophones([1,1,0,0])
hydrophone_array.print_hydrophone_toas()