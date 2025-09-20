import pytest
import utils.util_funcs as ut
from utils.wfm import WfmSquare, FilterButterworth

@pytest.mark.parametrize("freq_Hz, duration_s, period_std_n", [
    (880, 1, 5),
    (880.5, 4, 5),
    (880, 0.5, 5),
    (881, 1, 5),
    (881, 1, 0),
])
def test_create_square_wave_with_guassian(freq_Hz, duration_s, period_std_n):
    
    print(f"\n\nRunning case f={freq_Hz}Hz, d={duration_s}s, and std_T_n={period_std_n}samples")
            
    sample_rate_Hz = 192000
    period_std_s = period_std_n / sample_rate_Hz
    
    # Create waveform
    wfm = WfmSquare(freq_Hz, duration_s, period_std_s)
    wfm.add_filter_to_list(FilterButterworth(wfm.sample_rate_Hz, 10000, 10, "low"))
    wfm.create_wfm()
    
    # Perform time domain checks
    print(f"==Time domain checks==")
    mean, std, count = ut.analyze_transitions(wfm.wfm, 0.5, False)
    freq_td_Hz = sample_rate_Hz / mean
    print(f"  Expected mean (in samples)={sample_rate_Hz/freq_Hz}; measured={mean}")
    print(f"  Expected std (in samples)={period_std_n}; measured={std}")
    print(f"  Expected count (in periods)={freq_Hz * duration_s}; measured={count}")
    print(f"  Expected freq_td_Hz (Hz)={freq_Hz}; measured={freq_td_Hz}")
    assert mean == pytest.approx(sample_rate_Hz / freq_Hz, abs=0.01*sample_rate_Hz/freq_Hz)
    assert std == pytest.approx(period_std_n, abs=0.1*period_std_n)
    assert count == pytest.approx(freq_Hz * duration_s, abs=4*duration_s)
    assert freq_td_Hz == pytest.approx(freq_Hz, abs=0.01*freq_Hz)
        
    # Perform frequency domain checks
    print(f"==Frequency domain checks==")
    freq_fft_Hz = ut.calculate_fundamental_frequency(wfm.wfm, sample_rate_Hz)
    print(f"  Expected fft_freq (Hz)={freq_Hz}; measured={freq_fft_Hz}, resolution={1/duration_s}Hz")
    assert freq_fft_Hz == pytest.approx(freq_Hz, abs=0.01*freq_Hz)
    
    
def test_create_wav_filter_squarewave_file(filepath: str, sample_rate_Hz:int, freq_Hz:float, duration_s:float):
    
    # Create a waveform
    b,a = ut.create_filter_butterworth(bw_filt_Hz=10E3, sample_rate_Hz=sample_rate_Hz, order=10)
    wfm = ut.create_filtered_square_wave_with_guassian(freq_Hz, duration_s, sample_rate_Hz, period_std_s, (b,a))
    
    