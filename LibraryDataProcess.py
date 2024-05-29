from scipy.signal import butter, filtfilt
import numpy as np

# ButterWorth濾波
def filter_2sIIR(sig, f, fs, n, type='low'):
    sigfilter = np.zeros_like(sig)
    b, a = butter(n, f / (fs / 2), type)
    sigfilter = filtfilt(b, a, sig)
    return sigfilter

# 最大最小歸一化(將信號壓縮到[0,1])
def maxmin_normalize_signal(origin):
    signal = np.copy(origin)
    max_sig = np.max(signal)
    min_sig = np.min(signal)
    signal = (signal-min_sig)/(max_sig-min_sig)
    return signal

# 儲存適合波段
def fing_good_signal_segment(bad_ppg,continuous_good_signal,fs):
    good_ppg = []
    for i in range(len(bad_ppg)//2):
        start_idx = bad_ppg[i*2]
        end_idx = bad_ppg[i*2+1]
        if end_idx - start_idx >= continuous_good_signal*fs:
            good_ppg.append([start_idx,end_idx])
    return good_ppg