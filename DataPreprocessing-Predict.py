import numpy as np
from LibraryDataProcess import filter_2sIIR, fing_good_signal_segment, maxmin_normalize_signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import sys
import json
import tempfile

'''
步驟
1.GR修正往後3個點(因硬體延遲造成)
2.high pass filter
3.取baseline = raw - high pass, DC = mean(baseline)
4.切除不良訊號， 分割成windows
5.取得每個windows的DC值
6.low pass filter
7.以心臟週期切分成segments
8.SQI
9.以segments為單位做歸一化
10.取出N值(HR)
11.以segments為單位做插值法
'''
# 作為ML的Inputs
Inputs_IR = []
Inputs_RED = []
Inputs_GR = []

# Variable setting
fs = 200  # Sampling frequency
highpass = 0.5
lowpass = 10
order_low = 10
order_high = 6

# 從臨時文件中讀取數據
userinfo_path = sys.argv[1]
IR_path = sys.argv[2]
RED_path = sys.argv[3]
GR_path = sys.argv[4]

with open(userinfo_path, 'r') as f:
    userinfo_json = f.read()
with open(IR_path, 'r') as f:
    IR_json = f.read()
with open(RED_path, 'r') as f:
    RED_json = f.read()
with open(GR_path, 'r') as f:
    GR_json = f.read()

userinfo = json.loads(userinfo_json)
IR = json.loads(IR_json)
RED = json.loads(RED_json)
GR = json.loads(GR_json)

# 取Baseline(去除過大的人為晃動)
highpass_RED = filter_2sIIR(RED, highpass, fs, order_high, 'high')
baseline_RED = RED - highpass_RED
highpass_IR = filter_2sIIR(IR, highpass, fs, order_high, 'high')
baseline_IR = IR - highpass_IR
highpass_GR = filter_2sIIR(GR, highpass, fs, order_high, 'high')
baseline_GR = GR - highpass_GR

# 找不適合的波段(以IR為指標)[[bad1_start,bad1_end],[bad2_start,bad2_end]...]
Mask_width = 500
step = 200
num_steps = (len(baseline_IR) - Mask_width) // step + 1
bad_windows = [0]
previous_mask_sqi = 1
for i in range(num_steps):
    start_idx = i * step
    end_idx = start_idx + Mask_width
    current_mask = baseline_IR[start_idx:end_idx]
    current_mask_std = np.std(current_mask)
    if current_mask_std>4000:
        if previous_mask_sqi == 1:
            bad_windows.append(end_idx - step*3)
        previous_mask_sqi = 0
    else:
        if previous_mask_sqi == 0:
            bad_windows.append(start_idx + step*2)
        previous_mask_sqi = 1
bad_windows.append(len(baseline_IR))

# 取得DC值
good_windows = fing_good_signal_segment(bad_windows,10,fs)
DC = [] # [[IR,RED,GR],...]
for i in range(len(good_windows)):
    DC_IR = np.mean(baseline_IR[good_windows[i][0]:good_windows[i][1]])
    DC_RED = np.mean(baseline_RED[good_windows[i][0]:good_windows[i][1]])
    DC_GR = np.mean(baseline_GR[good_windows[i][0]:good_windows[i][1]])
    DC.append([DC_IR,DC_RED,DC_GR])

# low pass
filtered = []
filtered.append(filter_2sIIR(highpass_IR, lowpass, fs, order_low, 'low'))
filtered.append(filter_2sIIR(highpass_RED, lowpass, fs, order_low, 'low'))
filtered.append(filter_2sIIR(highpass_GR, lowpass, fs, order_low, 'low'))

# 以心臟週期切割
good_segments = []
bad_segments = []
ng = 0
nb = 0
for j in range(len(good_windows)):
    peaks, _ = find_peaks(filtered[0][good_windows[j][0]:good_windows[j][1]], distance=100)
    # sqi
    Bpeaks=[]
    Gpeaks=[]
    for i in range(len(peaks)-1):
        # 正常人休息心律60~100bpm,以30~240bpm算,間隔要在50~400
        # |start-end|<k*height
        peaki = peaks[i]+good_windows[j][0]
        peakf = peaks[i+1]+good_windows[j][0]
        interval = peaks[i+1] - peaks[i]
        ppg_max = max(filtered[0][peaki:peakf])
        ppg_min = min(filtered[0][peaki:peakf])
        ppg_start = filtered[0][peaki]
        ppg_end = filtered[0][peakf]
        height = abs(ppg_max-ppg_min)

        if interval > 400 or interval < 50 or abs(ppg_start-ppg_end)>0.2*height:
            Bpeaks.append([peaki,peakf])
            nb+=1
        else:
            Gpeaks.append([peaki,peakf])
            ng+=1
    good_segments.append(Gpeaks)
    bad_segments.append(Bpeaks)

# 歸一化與插值法
for c in range(3):
    processed_segments = []
    for i in range(len(good_segments)):
        for j in range(len(good_segments[i])):
            # 取出AC值
            AC = max(filtered[c][good_segments[i][j][0]:good_segments[i][j][1]])-min(filtered[c][good_segments[i][j][0]:good_segments[i][j][1]])
            # 最大最小歸一化
            normalized_segment = maxmin_normalize_signal(filtered[c][good_segments[i][j][0]:good_segments[i][j][1]])
            # 取出HR(N Point)
            N = len(normalized_segment)
            # 三次樣條插值法
            x = np.linspace(0, len(normalized_segment), num=len(normalized_segment))
            f = interp1d(x, normalized_segment, kind='cubic')
            xnew = np.linspace(0, len(normalized_segment), num=200)
            interpolated_segment = f(xnew)

            combine = [N, AC, DC[i][c]] + list(interpolated_segment)
            processed_segments.append(combine)
    
    # 儲存
    if c == 0:
        Inputs_IR = Inputs_IR + processed_segments
    elif c == 1:
        Inputs_RED = Inputs_RED + processed_segments
    else:
        Inputs_GR = Inputs_GR + processed_segments

with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f_IR, \
     tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f_RED, \
     tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f_GR:

    json.dump(Inputs_IR, f_IR)
    json.dump(Inputs_RED, f_RED)
    json.dump(Inputs_GR, f_GR)

    IR_output_path = f_IR.name
    RED_output_path = f_RED.name
    GR_output_path = f_GR.name

print(IR_output_path)
print(RED_output_path)
print(GR_output_path)