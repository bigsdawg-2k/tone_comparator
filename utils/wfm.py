import math, os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
import wave

@dataclass
class FilterButterworth:
    sample_rate_Hz: int
    cutoff_Hz:      float
    order:          int
    type:           str
    
class Wfm(ABC):
    
    C_DEFAULT_SR_Hz = 192000
    
    def __init__(self, freq_Hz:float=None, duration_s:float=None, sample_rate_Hz:int=None, filepath:str=None):
        
        if not filepath and (not freq_Hz and not duration_s):
            raise ValueError(f'You must specify at least one of (filepath) or (freq_Hz and duration_s)')
        
        self._filter_list = []
        self.wfm = []
        
        if filepath:
            self._read_wave(filepath)
        else:        
            self.freq_Hz = freq_Hz
            self.duration_s = duration_s
            if freq_Hz:
                self.sample_rate_Hz = sample_rate_Hz or Wfm.C_DEFAULT_SR_Hz
        
    def add_filter_to_list(self, filter_settings:FilterButterworth):
        
        if isinstance(filter_settings, FilterButterworth):
            f_nyquist_Hz = filter_settings.sample_rate_Hz / 2
            normalized_cutoff_Hz = filter_settings.cutoff_Hz / f_nyquist_Hz
            b,a = butter(filter_settings.order, normalized_cutoff_Hz, btype=filter_settings.type)
            self._filter_list.append([filter_settings, [b,a]])
        else:
            raise ValueError(f'Invalid filter settings for filter_type')
    
    def create_wfm(self):
        
        self._create_wfm()
        self._apply_filters()
    
    def write_wave(self, filepath:str):
        
        # TODO, Make this a little more dynamic...
        bits_n = 16
        scale = 0.8 * (((2**bits_n)/2)-1) / max(self.wfm) # 80% of full-scale
        wfm_scaled = (self.wfm * scale).astype(np.int16)
        with wave.open(filepath, mode='wb') as wave_write:
            wave_write.setframerate(self.sample_rate_Hz)
            wave_write.setnchannels(1)
            wave_write.setsampwidth(int(bits_n/8)) 
            wave_write.writeframes(wfm_scaled.tobytes())
        
    def _apply_filters(self):
        
        for filter_list_item in self._filter_list:
            orig_filter_settings = filter_list_item[0]
            if isinstance(orig_filter_settings, FilterButterworth):
                self.wfm = filtfilt(filter_list_item[1][0], filter_list_item[1][1], self.wfm)        

    def _read_wave(self, filepath: str):
        
        with wave.open(filepath, mode='rb') as wave_read:
            self.sample_rate_Hz = wave_read.getframerate()
            n_channels = wave_read.getnchannels()
            sampwidth = wave_read.getsampwidth()
            n_frames = wave_read.getnframes()
            audio_bytes = wave_read.readframes(n_frames)

        # Determine dtype
        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=dtype)

        # If multi-channel, reshape and average to mono
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = audio.mean(axis=1).astype(dtype)

        self.wfm = audio.astype(np.float32) / np.iinfo(dtype).max  # Normalize to [-1, 1] float
            
    @abstractmethod
    def _create_wfm(self):
        raise NotImplementedError(f'_create_wfm not implemented in Wfm')

class WfmSquare(Wfm):
    
    def __init__(self, freq_Hz:float=None, duration_s:float=None, period_std_s:float=None, sample_rate_Hz:int=None, filepath:str=None):
        
        # Sanitize arguments
        if filepath is not None and (period_std_s and freq_Hz) and period_std_s > 0.25 * 1 / freq_Hz:
            raise ValueError(f'Period standard deviation exceeds 25% of full period.')
        
        super().__init__(freq_Hz, duration_s, sample_rate_Hz, filepath)
        self.period_std_s = period_std_s
    
    def _create_wfm(self):
        
        # Generate a ndarray of floats of the period durations in seconds, for each period, following a normal 
        # distribution.
        duration_n = math.ceil(self.duration_s * self.sample_rate_Hz)
        rand_gen = np.random.default_rng()
        period_std_n = self.period_std_s * self.sample_rate_Hz
        period_n = self.sample_rate_Hz / self.freq_Hz # Keep these fractional, as part of normal dist to round.
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

        self.wfm = wfm

