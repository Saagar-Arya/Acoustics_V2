import array
from dataclasses import dataclass
import struct
import sys
import numpy as np
import pandas as pd

TYPE_DIGITAL = 0
TYPE_ANALOG = 1
expected_version = 0

@dataclass
class AnalogData:
    begin_time: float
    sample_rate: int
    downsample: int
    num_samples: int
    samples: np.ndarray

def parse_analog_binary(filename: str) -> AnalogData:
    f = open(filename, 'rb')
    # Parse header
    identifier = f.read(8)
    if identifier != b"<SALEAE>":
        raise Exception("Not a saleae file")

    version, datatype = struct.unpack('<ii', f.read(8))

    if version != expected_version or datatype != TYPE_ANALOG:
        raise Exception("Unexpected data type: {}".format(datatype))

    # Parse analog-specific data
    begin_time, sample_rate, downsample, num_samples = struct.unpack('<dqqq', f.read(32))

    f.close()

    # Parse samples
    samples = array.array("f")
    samples.fromfile(f, num_samples)

    return AnalogData(begin_time, sample_rate, downsample, num_samples, np.asarray(samples))

def parse_analog_csv(filename: str) -> list[AnalogData]:
    skip_rows = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            try:
                float(parts[0])
                skip_rows = i
                break
            except ValueError:
                continue

    data = pd.read_csv(filename, skiprows=skip_rows, header=None)
    
    times = data.iloc[:, 0].to_numpy()
    num_hydrophones = data.shape[1] - 1
    datas = []
    for idx in range (0,num_hydrophones):
        samples = data.iloc[:, idx+1].to_numpy()
        begin_time = times[0]
        sample_rate = 1.0 / (times[1] - times[0]) if len(times) > 1 else 0
        downsample = 1
        num_samples = len(samples)
        datas.append(AnalogData(begin_time, sample_rate, downsample, num_samples, samples))
    return datas


if __name__ == '__main__':
    filename = sys.argv[1]
    print("Opening " + filename)

    data = parse_analog_binary(filename)

    # Print out all analog data
    print("Begin time: {}".format(data.begin_time))
    print("Sample rate: {}".format(data.sample_rate))
    print("Downsample: {}".format(data.downsample))
    print("Number of samples: {}".format(data.num_samples))

    print("  {0:>20} {1:>10}".format("Time", "Voltage"))

    for idx, voltage in enumerate(data.samples):
        sample_num = idx * data.downsample
        time = data.begin_time + (float(sample_num) / data.sample_rate)
        print("  {0:>20.10f} {1:>10.3f}".format(time, voltage))