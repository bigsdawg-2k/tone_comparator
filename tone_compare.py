import os, time, yaml, multiprocessing, keyboard
import sounddevice as sd
from typing import List, Union, Any
import numpy as np
import scipy.signal as signal
from scipy.io.wavfile import write
import utils.util_funcs as ut
from utils.wfm import WfmSquare, FilterButterworth

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
            parsed.append([item[0], [float(item[1]), float(item[2]), float(item[3]), float(item[4])]])
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
            m = int(user_input)
            if 0 <= m < len(devices):
                selected_indexes.append(m)
                if isinstance(devices[m], dict):
                    print(f"Selected: {devices[m]['name']}")
                else:
                    print(f"Selected: {devices[m][0]} {devices[m][1]}")
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
    
    print(f'\nDuration is set to {duration_s}s.  \nFrequency resolution limited to {1/duration_s:.3f}Hz\n')

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
        m = 0
        last_dev_freq_Hz = None
        while True:
            
            this_dev = devices[selected_indexes[m]]
            
            # Handle sound device case
            if isinstance(this_dev, dict) and 'hostapi' in this_dev.keys():
                print(f'Sound device: {this_dev['name']}. {duration_s}s@{sample_rate_Hz}Hz')
                sd.default.device = (this_dev['index'], None)
                audio = sd.rec(int(duration_s * sample_rate_Hz), samplerate=sample_rate_Hz, channels=this_dev['max_input_channels'], dtype='float32')
                sd.wait()
                # Convert to mono if needed
                if audio.ndim > 1 and audio.shape[1] > 1:
                    wfm_data = np.mean(audio, axis=1)
                else:
                    wfm_data = audio.flatten()
                sr_Hz = sample_rate_Hz
            
            # Handle the waveform case
            elif isinstance(this_dev, list) and this_dev[0] == 'square':
                print(f'Generating waveform: {this_dev}')
                wfm = WfmSquare(this_dev[1][0], this_dev[1][1], this_dev[1][3], sample_rate_Hz=this_dev[1][2])
                wfm.add_filter_to_list(FilterButterworth(wfm.sample_rate_Hz, 10000, 10, "low"))
                wfm.create_wfm()
                wfm_data = wfm.wfm
                sr_Hz = wfm.sample_rate_Hz
                time.sleep(len(wfm_data)/sr_Hz)
                                
            # Handle the filename case
            elif isinstance(this_dev, list) and this_dev[0] == 'file':
                print(f'File: {this_dev}')
                wfm = WfmSquare(filepath=this_dev[1])
                wfm_data = wfm.wfm
                sr_Hz = wfm.sample_rate_Hz
                time.sleep(len(wfm_data)/sr_Hz)
                
            # Otherwise
            else:
                raise ValueError(f'Unexpected device: {this_dev}')

            freq_Hz = ut.calculate_fundamental_frequency(wfm_data, sr_Hz)
            
            # Calculate ratio if there is a previous result then print result
            if last_dev_freq_Hz is not None:
                str_compare = f', Ratio={freq_Hz/last_dev_freq_Hz:.3f}'
            else:
                str_compare = ''
            print(f'Freq={freq_Hz}Hz{str_compare}')

            # Look for space to advance
            while not keypress_queue.empty():
                msg = keypress_queue.get()
                if msg == 'space':
                    m = (m + 1) % len(selected_indexes)
                    last_dev_freq_Hz = freq_Hz
            
    except KeyboardInterrupt:
        print("\nStopped.")
    
    # Terminate background process
    finally:
        if 'keypress_proc' in locals() and keypress_proc.is_alive():
            keypress_proc.terminate()
            keypress_proc.join()
        
if __name__ == "__main__":
    main()