import pytest, os
import utils.util_funcs as ut
from utils.wfm import Wfm, WfmSquare, FilterButterworth

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
    
@pytest.mark.parametrize("filepath, freq_Hz, duration_s, sample_rate_Hz", [
    ('test_wav_del_me.wav', 440, 1, 192000),
    ('test_wav_del_me.wav', 440, 1, 44100)
])
def test_create_write_wav(filepath: str, freq_Hz:float, duration_s:float, sample_rate_Hz:int):
    
    C_PERIOD_STD_s = 0.01/1E3
    
    if os.path.exists(filepath):
        os.remove(filepath)        
    assert os.path.exists(filepath) == False
    
    # Create a waveform and then save it.
    wfm = WfmSquare(freq_Hz, duration_s, C_PERIOD_STD_s, sample_rate_Hz)
    wfm.create_wfm()
    wfm.write_wave(filepath)
    
    assert os.path.exists(filepath) == True
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
@pytest.mark.parametrize("freq_Hz, duration_s", [
    (880, 1)
])
def test_create_wfm_from_file(freq_Hz:float, duration_s:float):
    
    C_FILEPATH_TEMP = 'test_wave_del_me.wav'
    C_PERIOD_STD_s = 0.01/1E3
    filepath = C_FILEPATH_TEMP
    
    if os.path.exists(filepath):
        os.remove(filepath)        
    assert os.path.exists(filepath) == False
    
    # Create a waveform and then save it.
    wfm = WfmSquare(freq_Hz, duration_s, C_PERIOD_STD_s)
    wfm.create_wfm()
    wfm.write_wave(filepath)
    
    assert os.path.exists(filepath) == True
    
    wfm2 = WfmSquare(filepath=filepath)
    assert wfm2.sample_rate_Hz == WfmSquare.C_DEFAULT_SR_Hz
        
    if os.path.exists(filepath):
        os.remove(filepath)
    