
from Scripts import Hydrophone_Array
import matplotlib.pyplot as plt

from Scripts.Hydrophone import Hydrophone

def plot_selected_hydrophones(hydro_array: Hydrophone_Array, selected: list[bool]):
    subplots = sum(selected)
    fig, axes = plt.subplots(subplots, 1, figsize=(10, 10), sharex=True)

    plot = 0
    for hydro, s in zip(hydro_array.hydrophones, selected):
        if s:
            hydro_array.plot_hydrophone(hydro, axes[plot])
            plot+=1
        
    plt.tight_layout()
    plt.show()


def plot_envelope_hydrophone(hydrophone: Hydrophone, ax):
    if hydrophone.toa.found_peak is False:
        ax.text(0.5, 0.5, "No data loaded", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("ToA Detection (No Data)")
        ax.axis("off")
        return
    
    ax.plot(hydrophone.times, hydrophone.voltages, label="Original")
    if getattr(hydrophone, "filtered_signal", None) is not None:
        ax.plot(hydrophone.times, hydrophone.filtered_signal, label="Filtered")
    if getattr(hydrophone, "envelope", None) is not None:
        ax.plot(hydrophone.times, hydrophone.toa.envelope, label="Envelope", linestyle="--")
    if getattr(hydrophone, "toa_time", None) is not None:
        ax.axvline(hydrophone.toa.toa_time, color="r", linestyle=":", label=f"ToA = {hydrophone.toa.toa_time:.6f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage")
    ax.set_title("ToA Detection")
    ax.legend(loc="best")
    ax.grid(True)