import pytest
import utils.util_funcs as ut

def test_create_square_wave_with_guassian():
    
    freq_Hz = 880 # A5
    duration_s = 1
    sample_rate_Hz = 192000
    period_std_s = 5 / sample_rate_Hz
    period_std_n = period_std_s * sample_rate_Hz
    
    # Create a waveform
    b,a = ut.create_filter_butterworth(bw_filt_Hz=18E3, sample_rate_Hz=sample_rate_Hz, order=5)
    wfm = ut.create_filtered_square_wave_with_guassian(freq_Hz, duration_s, sample_rate_Hz, period_std_s, (b,a))
    
    # Perform time domain checks
    mean, std, count = ut.analyze_transitions(wfm, 0.5, False)
    assert mean == pytest.approx(sample_rate_Hz / freq_Hz, abs=0.01*sample_rate_Hz/freq_Hz)
    assert std == pytest.approx(period_std_n, abs=0.05*period_std_n)
    assert count == pytest.approx(freq_Hz * duration_s, abs=1)
    print(f"Expected mean (in samples)={sample_rate_Hz/freq_Hz}; measured={mean}")
    print(f"Expected std (in samples)={period_std_n}; measured={std}")
    print(f"Expected count (in periods)={freq_Hz * duration_s}; measured={count}")
    
    # Perform frequency domain checks
    freq_fft_Hz = ut.calculate_fundamental_frequency(wfm, sample_rate_Hz)
    assert freq_fft_Hz == pytest.approx(freq_Hz, abs=0.01*freq_Hz)
    print(f"Expected fft_freq (Hz)={freq_Hz}; measured={freq_fft_Hz}")