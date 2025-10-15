import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_TOA as HydrophoneTOA

# folder = time.strftime('%Y-%m-%d--%H-%M-%S')
# path = os.path.join("Scripts",folder)
# os.mkdir(path)

# logic = LOGIC.Logic()
# logic.print_saleae_status()
# csv_path = logic.start_capture(1.5,path)
# logic.kill_logic()
# plt.figure()

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