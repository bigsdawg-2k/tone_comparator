import os, sys, math
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple

def analyze_transitions(wfm: np.ndarray, threshold: float, pos_edge: bool) -> tuple[float, float, int]:
    """Analyze waveform transitions across a threshold.

    Args:
        wfm (np.ndarray): Waveform data.
        threshold (float): Transition threshold.
        pos_edge (bool): If True, count positive-going transitions; else negative-going

    Returns:
        tuple[float, float, int]: 
            1) mean: Mean number of samples between transitions,
            2) std: Standard deviation of number of samples between transitions,
            3) count: Number of transitions.
    """    
    
    # Create a boolean mask for threshold crossing
    if pos_edge:
        crossings = (wfm[:-1] < threshold) & (wfm[1:] >= threshold)
    else:
        crossings = (wfm[:-1] > threshold) & (wfm[1:] <= threshold)

    # Get indices where transitions occur
    transition_indices = np.where(crossings)[0] + 1  # +1 to get the index of the crossing point

    # Calculate durations between transitions
    if len(transition_indices) >= 2:
        durations = np.diff(transition_indices)
        mean = float(np.mean(durations))
        std = float(np.std(durations))
    else:
        durations = np.array([])
        mean = 0.0
        std = 0.0

    count = len(transition_indices)

    return mean, std, count

def calculate_fundamental_frequency(wfm: np.ndarray, sample_rate_Hz: float) -> float:
    """
    Estimate the fundamental frequency of a waveform using FFT.

    Args:
        wfm (np.ndarray): Input waveform.
        sample_rate_Hz (float): Sample rate.

    Returns:
        float: Fundamental frequency in Hz
    """    
    
    # Compute FFT and frequency bins
    fft = np.fft.rfft(wfm)
    freqs = np.fft.rfftfreq(len(wfm), d=1/sample_rate_Hz)
    magnitude = np.abs(fft)

    # Ignore DC component
    magnitude[0] = 0

    # Find index of peak magnitude
    peak_index = np.argmax(magnitude)
    freq_Hz = freqs[peak_index]

    return float(freq_Hz)

def create_filter_butterworth(bw_filt_Hz:float, sample_rate_Hz:float, order:int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Butterworth low-pass filter.

    Parameters:
        cutoff (float): Cutoff frequency in Hz.
        fs (int): Sampling rate in Hz.
        order (int): Filter order.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filter coefficients (b, a).
    """
    f_nyquist_Hz = sample_rate_Hz/2
    normalized_cutoff_Hz = bw_filt_Hz/f_nyquist_Hz
    b,a = butter(order, normalized_cutoff_Hz, btype="low")
    return b,a

def create_filtered_square_wave_with_guassian(freq_Hz:float, duration_s:float, sample_rate_Hz:float, period_std_s:float, filt_coefs:Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Create a square wave wfm of a given frequency, duration, and sample rate while having a period that varies with 
    a standard deviation.
    
    Applies a LPF to limit signal bandwidth to audio range.

    Args:
        freq_Hz (float): Nominal frequency of the waveform.
        duration_s (float): Duration of the waveform in seconds.
        sample_rate_Hz (float): Sample rate of the waveform in Hz.
        period_std_s (float): Standard deviation of the period length is s.

    Returns:
        np.ndarray: Array of wfm samples
    """
    # Determine the period length in samples
    period_n = sample_rate_Hz / freq_Hz
    if period_n == int(period_n):
        period_n = int(period_n)
    else:
        print(f"Detected non-integer period ({period_n}), rounding down to nearest full sample {math.floor(period_n)}.")
        period_n = math.floor(period_n)
    
    # Generate a ndarray of floats of the period durations in seconds, for each period, following a normal 
    # distribution.
    duration_n = math.ceil(duration_s * sample_rate_Hz)
    rand_gen = np.random.default_rng()
    period_std_n = period_std_s * sample_rate_Hz
    periods_n = rand_gen.normal(period_n, period_std_n, math.ceil(duration_n / period_n)).round().astype(int)

    # Create waveform array
    n_samples_half_mean_period = math.floor(period_n / 2)
    wfm = np.zeros(duration_n)
    
    # Iterate over all the randomly distributed period lengths and make the waveform by
    # concatenating the periods.  Always start off (0) and remain off for 50% of mean 
    # period time, take up rest of period on (1.0).  Initialized array is already at 0
    # so just need to set ones.
    idx = 0
    for this_period_n in periods_n:
        if (slice_end := idx + this_period_n) > len(wfm):
            slice_end = len(wfm)
        wfm[idx + n_samples_half_mean_period:slice_end] = 1.0
        idx = idx + this_period_n

    # Apply low-pass filter
    wfm_filtered = filtfilt(filt_coefs[0], filt_coefs[1], wfm)
           
    return wfm_filtered

def parse_named_args() -> dict:
    """
    Parses command-line arguments of the form:
        --argName value {}'argName': 'value'}
        -flagName       {'flagName': True}
        
    If --argName is specified without a following value then it will be assigned None

    Returns:
        dict: Dictionary of parsed arguments.
    """
    
    args = sys.argv[1:]  # Skip script name
    args_parsed = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if arg.startswith('--'):
            key = arg[2:]
            # Check if next item exists and isn't another flag
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                args_parsed[key] = args[i + 1]
                i += 2
            else:
                args_parsed[key] = None
                i += 1

        elif arg.startswith('-'):
            key = arg[1:]
            args_parsed[key] = True
            i += 1

        else:
            # Positional or unexpected argument, skip.
            i += 1  

    return args_parsed
