import os, time, yaml, multiprocessing, keyboard
import sounddevice as sd
from typing import List, Union, Any
import numpy as np
import scipy.signal as signal
from scipy.io.wavfile import write
import utils.util_funcs as ut

def input_devices_as_list() -> list:
    """
    List the available input devices.

    Returns:
        sd.DeviceList: List of input devices.
    """
    
    return list(sd.query_devices())

def parse_debug_waveforms(debug_waveforms:dict, ) -> list[str, list]:

    parsed = []
    for item in debug_waveforms:
        if isinstance(item, str):
            parsed.append(['file', item])
        elif isinstance(item, list):
            parsed.append([item[0], [int(item[1]), float(item[2]), int(item[3])]])
        else:
            raise TypeError(f'Unexpected debug waveform format.')
        
    return parsed    
        
def select_devices(devices:List[Union[sd.DeviceList, dict, str]]) -> List[int]:
    """
    Allow user to select multiple device indexes, one per line. Empty input to finish.
    Returns a list of selected indexes.

    Args:
        devices (List[Union[sd.DeviceList], dict, str]): Devices (and debug) to choose from.

    Returns:
        List[int]: List of selected indexes.
    """    
    selected_indexes = []
    while True:
        user_input = input("\nEnter device index (empty to finish): ").strip()
        if user_input == "":
            break
        try:
            idx = int(user_input)
            if 0 <= idx < len(devices):
                selected_indexes.append(idx)
                if isinstance(devices[idx], dict):
                    print(f"Selected: {devices[idx]['name']}")
                else:
                    print(f"Selected: {devices[idx][0]} {devices[idx][1]}")
            else:
                print("Index out of range.")
        except ValueError:
            print("Invalid input. Please enter a number or empty to finish.")
    return selected_indexes

def get_fft_peak_ratio(audio, sample_rate_Hz, freqs):
    # FFT
    fft_data = np.fft.rfft(audio)
    fft_freqs = np.fft.rfftfreq(len(audio), 1/sample_rate_Hz)
    magnitudes = np.abs(fft_data)

    # Find closest bins
    bins = [np.argmin(np.abs(fft_freqs - f)) for f in freqs]
    values = [magnitudes[b] for b in bins]
    ratio = values[0] / values[1] if values[1] != 0 else np.inf
    return values, ratio

def get_time_domain_ratio(audio, sample_rate_Hz, freqs):
    # Bandpass filters
    values = []
    for f in freqs:
        band = [f - 10, f + 10]
        sos = signal.butter(4, band, btype='bandpass', fs=sample_rate_Hz, output='sos')
        filtered = signal.sosfilt(sos, audio)
        amplitude = np.sqrt(np.mean(filtered**2))  # RMS
        values.append(amplitude)
    ratio = values[0] / values[1] if values[1] != 0 else np.inf
    return values, ratio

def keypress_monitor(q, ready_event:Any):
    ready_event.set()
    while True:
        if keyboard.is_pressed('space'):
            q.put('space')
            while keyboard.is_pressed('space'):
                time.sleep(0.005)  # Debounce
        time.sleep(0.01)

def main():

    # Handle command line argument(s)
    parsed_args = ut.parse_named_args()
    filename_cfg = parsed_args.get('cfgFile', os.path.join('.', 'config.yaml'))

    # Parse config file and load values
    with open(filename_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        
    duration_s = cfg['duration_s']
    sample_rate_Hz = cfg['sample_rate_Hz']

    # Get devices and debug settings
    devices = input_devices_as_list()
    devices.extend(parse_debug_waveforms(cfg['debug']['waveforms']))
    
    # List all the things...
    print("Available input devices:")
    for i, dev in enumerate(devices):
        if isinstance(dev, dict):
            name = f'{dev['name']}'
        else:
            name = f'{dev[0]} {dev[1]}'
        print(f"{i}: {name}")
    
    # Allow user selection
    selected_indexes = select_devices(devices)
    
    # Spawn background process to watch for keypresses
    keypress_queue = multiprocessing.Queue()
    ready_event = multiprocessing.Event()
    keypress_proc = multiprocessing.Process(target=keypress_monitor, args=(keypress_queue, ready_event), daemon=True)
    keypress_proc.start()
    ready_event.wait()
    
    print("\nPress Ctrl+C to stop.\n")
    try:
        idx = 0
        while True:
            # audio = sd.rec(int(duration_s * sample_rate_Hz), samplerate=sample_rate_Hz, channels=1, dtype='float32')
            # sd.wait()
            # audio = audio.flatten()

            # fft_vals, fft_ratio = get_fft_peak_ratio(audio, sample_rate_Hz, target_freqs)
            # time_vals, time_ratio = get_time_domain_ratio(audio, sample_rate_Hz, target_freqs)

            # print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            # print(f"FFT Magnitudes: {fft_vals}, Ratio: {fft_ratio:.2f}")
            # print(f"Time-Domain RMS: {time_vals}, Ratio: {time_ratio:.2f}")
            print(f'{idx}')

            time.sleep(0.5)
            
            while not keypress_queue.empty():
                msg = keypress_queue.get()
                if msg == 'space':
                    idx = (idx + 1) % len(selected_indexes)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    
    # Terminate background process
    finally:
        if 'keypress_proc' in locals() and keypress_proc.is_alive():
            keypress_proc.terminate()
            keypress_proc.join()
        
if __name__ == "__main__":
    main()