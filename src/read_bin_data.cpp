#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <cmath>

int main() {
    // ---- configuration ----
    const std::string path = "data/analog_3.bin";  // this file is ONE channel
    const double fs       = 781250.0;              // 781.25 kS/s
    const size_t headerBytes = 64;                 // skip <SALEAE> header
    const bool useFloat32    = true;               // true: float32 samples, false: int16_t

    // CSV starts at 75.122 µs and advances by 1/fs = 1.28 µs per row.
    const double t0_s    = 0.000075122;            // absolute time of sample index 0
    const double start_s = 0.0;            // print window start (absolute time)
    const double end_s   = 0.1;            // print window end   (absolute time)
    const size_t decimate = 1;                     // 1 -> every sample (Δt = 1/fs)

    // ---- open file ----
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Error: could not open " << path << "\n";
        std::cerr << "Press Enter to exit..."; std::cin.get();
        return 1;
    }
    std::streamsize bytes = f.tellg();
    if (bytes <= static_cast<std::streamsize>(headerBytes)) {
        std::cerr << "Error: file too small or header too large.\n";
        std::cerr << "Press Enter to exit..."; std::cin.get();
        return 1;
    }
    f.seekg(headerBytes); // skip header

    std::cout << std::fixed;
    std::cout << "Time [s],Channel 0\n";

    auto clamp_and_grid = [&](size_t &i0, size_t &i1, size_t n) -> bool {
        if (i0 >= n) i0 = (n ? n - 1 : 0);
        if (i1 >  n) i1 = n;
        if (i1 <= i0) return false;
        if (decimate > 1) i0 = (i0 + decimate - 1) / decimate * decimate; // snap to grid
        return true;
    };

    if (useFloat32) {
        // ---- read float32 samples ----
        const size_t n = static_cast<size_t>((bytes - headerBytes) / 4);
        std::vector<float> data(n);
        if (!f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 4))) {
            std::cerr << "Error: failed to read float32 samples.\n";
            std::cerr << "Press Enter to exit..."; std::cin.get();
            return 1;
        }

        // Convert absolute times to sample indices:  t = t0 + i/fs  =>  i = (t - t0)*fs
        size_t i0 = (start_s <= t0_s) ? 0 : static_cast<size_t>(std::llround((start_s - t0_s) * fs));
        size_t i1 = (end_s   <= t0_s) ? 0 : static_cast<size_t>(std::llround((end_s   - t0_s) * fs));
        if (!clamp_and_grid(i0, i1, n)) {
            std::cerr << "Selected time window has no samples.\n";
            std::cerr << "Press Enter to exit..."; std::cin.get();
            return 1;
        }

        for (size_t i = i0; i < i1; i += decimate) {
            const double t = t0_s + static_cast<double>(i) / fs; // print absolute time
            const double y = static_cast<double>(data[i]);
            std::cout << std::setprecision(9) << t << ','
                      << std::setprecision(3) << y << "\n";
        }

    } else {
        // ---- read int16 samples ----
        const size_t n = static_cast<size_t>((bytes - headerBytes) / 2);
        std::vector<int16_t> data(n);
        if (!f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 2))) {
            std::cerr << "Error: failed to read int16 samples.\n";
            std::cerr << "Press Enter to exit..."; std::cin.get();
            return 1;
        }

        size_t i0 = (start_s <= t0_s) ? 0 : static_cast<size_t>(std::llround((start_s - t0_s) * fs));
        size_t i1 = (end_s   <= t0_s) ? 0 : static_cast<size_t>(std::llround((end_s   - t0_s) * fs));
        if (!clamp_and_grid(i0, i1, n)) {
            std::cerr << "Selected time window has no samples.\n";
            std::cerr << "Press Enter to exit..."; std::cin.get();
            return 1;
        }

        for (size_t i = i0; i < i1; i += decimate) {
            const double t = t0_s + static_cast<double>(i) / fs;
            const double y = static_cast<double>(data[i]) / 32768.0;
            std::cout << std::setprecision(9) << t << ','
                      << std::setprecision(3) << y << "\n";
        }
    }

    // ---- pause before exit (so double-clicking shows output) ----
    std::cerr << "Done. Press Enter to exit...";
    std::cin.get();
    return 0;
}
