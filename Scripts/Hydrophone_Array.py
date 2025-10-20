from Hydrophone import Hydrophone, HydrophoneConfig
from read_files import parse_analog_csv

class Hydrophone_Array:
    def __init__(
        self, 
        search_band_min:float=25000,
        search_band_max:float=40000, 
        bandwidth:float=100.0,
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.bandwidth = float(bandwidth)

        self.hydrophone_0 = Hydrophone()
        self.hydrophone_1 = Hydrophone()
        self.hydrophone_2 = Hydrophone()
        self.hydrophone_3 = Hydrophone()
        self.hydrophones = [self.hydrophone_0,self.hydrophone_1,self.hydrophone_2,self.hydrophone_3]        

        self.threshold_factor = 0.3

    def set_hydrophone_config(self, configs: list[HydrophoneConfig]):
        for hydrophone, config in zip(self.hydrophones, configs):
            if config is not None:
                hydrophone.config = config
    
    def update_from_binaries(self, paths: list[str]):
        for hydrophone, path in zip(self.hydrophones, paths):
            hydrophone.update_from_binary(path)

    def update_from_csv(self, filename: str):
        datas = parse_analog_csv(filename)
        for hydrophone, data in zip(self.hydrophones, datas):
            hydrophone.update_from_data(data)

    def print_envelope_toas(self):
        for i, h in sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (not ih[1].toa.found_peak, ih[1].toa.toa_time if ih[1].toa.toa_time is not None else float('inf'))
        ):
            if h.toa_time is not None:
                print(f"Hydrophone {i} saw ping at {h.toa.toa_time:.6f}s (found_peak={h.toa.found_peak})")
            else:
                print(f"Hydrophone {i} saw ping at N/A (found_peak={h.toa.found_peak})")

    def bandpass_signal(self, hydrophone: Hydrophone) -> None:
        hydrophone.apply_bandpass(self.bandwidth, self.search_band_min, self.search_band_max)
                            
    def estimate_by_envelope(self, hydrophone: Hydrophone):
        hydrophone.estimate_by_envelope(self.bandwidth, self.search_band_min, self.search_band_max)

    def estimate_selected_by_envelope(self, selected: list[bool] = [True,True,True,True]):
        for hydrophone, s in zip(self.hydrophones, selected):
            if s:
                self.estimate_by_envelope(hydrophone)

    def gcc_phat(self, h0: Hydrophone, h1: Hydrophone, max_tau: float=None, regularization: float=1e-8, polarity_insensitive: bool=True):
        self.bandpass_signal(h0)
        self.bandpass_signal(h1)
        return h0.gcc_phat_against(h1, max_tau=max_tau, regularization=regularization, polarity_insensitive=polarity_insensitive)
    
    def estimate_selected_by_gcc(self, selected: list[bool] = [True, True, True, True], max_tau=None, regularization=1e-8, polarity_insensitive=True):
        """
        For each selected hydrophone (except hydrophone 0), estimate TDOA relative to hydrophone_0 using gcc_phat.
        Results are stored on each hydrophone object:
            hydrophone.tdoa_gcc (seconds)
            hydrophone.gcc_cc (cross-correlation array)
            hydrophone.gcc_lags (lags in seconds, aligned with gcc_cc)
            hydrophone.gcc_shift_samples (integer shift in samples)
        Hydrophone 0 will have tdoa_gcc = 0.0.
        """
        # ensure hydrophone_0 has data (and optionally precompute bandpass on it)
        if getattr(self.hydrophone_0, "voltages", None) is None:
            raise RuntimeError("Hydrophone 0 has no data. Load CSV first.")

        # compute bandpass for hydrophone 0 once (gcc_phat will call bandpass_signal internally too)
        # but call here explicitly to ensure peak_freq etc are set
        self.bandpass_signal(self.hydrophone_0)

        # set hydrophone 0 fields
        self.hydrophone_0.gcc_phat.set_as_reference()

        for idx, (hydro, sel) in enumerate(zip(self.hydrophones, selected)):
            if not sel:
                # skip unselected hydrophones
                continue

            if idx == 0:
                # already set above
                continue

            # call gcc_phat with hydrophone_0 as reference and hydro as other
            try:
                hydro.gcc_phat_against(reference=self.hydrophone_0)
            except Exception as e:
                # propagate useful info on failure but keep other hydrophones processed
                hydro.gcc_phat.reset()
                print(f"gcc_phat failed for hydrophone {idx}: {e}")
                continue

    def print_gcc_TDOA(self, selected: list[bool] = [True, True, True, True], indent: str = "  "):
        """
        Print TDOA results computed by estimate_selected_by_gcc for each hydrophone,
        always relative to hydrophone 0. If a hydrophone wasn't computed (tdoa_gcc is None)
        a 'N/A' is printed.
        """
        print("GCC-PHAT TDOA relative to Hydrophone 0")
        print("-" * 48)
        print(f"{'Hydrophone':<12}{'Selected':<10}{'TDOA (s)':<14}{'Shift (samples)':<16}{'Interpretation'}")
        print("-" * 48)

        for idx, (hydro, sel) in enumerate(zip(self.hydrophones, selected)):
            selected_str = "Yes" if sel else "No"
            tdoa = getattr(hydro, "tdoa_gcc", None)
            shift = getattr(hydro, "gcc_shift_samples", None)

            if tdoa is None:
                tdoa_str = "N/A"
            else:
                tdoa_str = f"{tdoa:+.6e}"  # show sign (+/-)

            if shift is None:
                shift_str = "N/A"
            else:
                shift_str = f"{shift}"

            # Interpretation: which hydrophone saw the ping first?
            if tdoa is None:
                interp = "no estimate"
            else:
                # tdoa = tau where positive means hydrophone is delayed relative to hydrophone_0 (hydrophone sees ping later)
                if idx == 0 or abs(tdoa) < 1e-12:
                    interp = "same time (reference)"
                elif tdoa > 0:
                    interp = f"Hydrophone 0 leads by {tdoa:.6e}s"
                else:
                    interp = f"Hydrophone {idx} leads by {abs(tdoa):.6e}s"

            print(f"{idx:<12}{selected_str:<10}{tdoa_str:<14}{shift_str:<16}{interp}")

        print("-" * 48)