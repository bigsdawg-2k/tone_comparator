import os, time, yaml
import sounddevice as sd
import numpy as np
import scipy.signal as signal
from scipy.io.wavfile import write
import utils.util_funcs as ut

# Handle command line argument(s)
parsed_args = ut.parse_named_args()
filename_cfg = parsed_args.get('cfgFile', os.path.join('.', 'config.yaml'))

# Parse config file and load values
with open(filename_cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    
duration_s = cfg['duration_s']
sample_rate_Hz = cfg['sample_rate_Hz']

target_freqs = [440, 880]  # Example: A4 and A5

def list_input_devices() -> sd.DeviceList:
    """
    List the available input devices.

    Returns:
        sd.DeviceList: List of input devices.
    """
    
    print("Available input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{i}: {dev['name']}")
    
    return devices

def list_wav_files(path_folder:str, idx_offset:int) -> list[str]:
    """
    List wav files in folder with a selectable index starting from idx_offset.

    Args:
        path_folder (str): _description_
        idx_offset (int): Starting offset for selectable index during printing.

    Returns:
        list[str]: Filenames of wav files in folder.
    """
    filenames = [f for f in os.listdir(path_folder) if f.lower().endswith('.wav')]
    for i in range(len(filenames)):
        print(f"{i + idx_offset}: {filenames[i]}")
    
    return filenames

def select_device(devices:sd.DeviceList):
    index = int(input("Select input device index: "))
    sd.default.device = (index, None)
    print(f"Using device: {devices[index]['name']}")

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

def main():

    # List devices and wav
    devices = list_input_devices()
    list_wav_files(cfg['path_wav_folder'], len(devices))
    
    select_device()
    print(f"Monitoring frequencies: {target_freqs[0]} Hz and {target_freqs[1]} Hz")
    print("Press Ctrl+C to stop.")

    audio_data = sd.rec(int(duration_s * sample_rate_Hz), samplerate=sample_rate_Hz, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished

    write("output.wav", sample_rate_Hz, audio_data)

    return

    try:
        while True:
            audio = sd.rec(int(duration_s * sample_rate_Hz), samplerate=sample_rate_Hz, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()

            fft_vals, fft_ratio = get_fft_peak_ratio(audio, sample_rate_Hz, target_freqs)
            time_vals, time_ratio = get_time_domain_ratio(audio, sample_rate_Hz, target_freqs)

            print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            print(f"FFT Magnitudes: {fft_vals}, Ratio: {fft_ratio:.2f}")
            print(f"Time-Domain RMS: {time_vals}, Ratio: {time_ratio:.2f}")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()