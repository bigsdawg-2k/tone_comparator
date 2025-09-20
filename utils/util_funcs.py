import sys, math
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from typing import Tuple
import matplotlib.pyplot as plt

# def analyze_transitions(wfm: np.ndarray, threshold: float, pos_edge: bool) -> tuple[float, float, int]:
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

    count = len(durations)

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
