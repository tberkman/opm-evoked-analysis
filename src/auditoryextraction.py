import scipy.io as sio, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
DATA_ROOT = r"C:\Users\Theod\Downloads\Megprojects\002"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)   
aud_run1 = os.path.join(DATA_ROOT, "OPM", "Auditory", "run01.meg.mat")
aud_run2 = os.path.join(DATA_ROOT, "OPM", "Auditory", "run02.meg.mat")
rest_run1 = os.path.join(DATA_ROOT, "OPM", "Rest", "run01.meg.mat")

def bandpass(x, lo, hi, sf, order=4):
    b, a = butter(order, [lo/(sf/2), hi/(sf/2)], btype="band")
    return filtfilt(b, a, x, axis=-1)

opm = sio.loadmat(aud_run1)
data = opm["bexp"]            # (30, 856000) field data, Tesla
trig = opm["bexp_ext"].ravel()# (856000,) trigger channel
sf   = 2000                   # Hz, from MEGinfo SampleFreq
data = bandpass(data, 1, 40, sf)   # apply to bexp BEFORE epoching

thr = (trig.max() + trig.min()) / 2         # halfway between flat and pulse
above = trig > thr
onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1   # low->high transitions
print(f"found {len(onsets)} tone onsets")
print("first few onset times (s):", np.round(onsets[:5]/sf, 2))

onsets = onsets + int(0.020 * sf)

tmin, tmax = -0.1, 0.4
n_pre, n_post = int(-tmin*sf), int(tmax*sf)
epochs = []
for o in onsets:
    if o+n_post < data.shape[1] and o-n_pre >= 0:
        seg = data[:, o-n_pre : o+n_post]        # (30, n_times)
        epochs.append(seg)
epochs = np.array(epochs)                          # (n_trials, 30, n_times)
print("epochs shape:", epochs.shape)

baseline = epochs[:, :, :n_pre].mean(axis=2, keepdims=True)
epochs = epochs - baseline
ptp = np.ptp(epochs, axis=2).max(axis=1)          # peak-to-peak per epoch, worst channel
good = ptp < np.percentile(ptp, 90)           # keep the cleanest 90%
print(f"keeping {good.sum()}/{len(good)} epochs")
evoked = epochs[good].mean(axis=0)
times = np.arange(-n_pre, n_post) / sf * 1000       # ms
peak_window = (times >= 80) & (times <= 130)
peak_amp = np.abs(evoked[:, peak_window]).max(axis=1)   # per-channel peak magnitude
order = np.argsort(peak_amp)[::-1]
print("channels with largest ~100ms response (index, magnitude fT):")
for i in order[:6]:
    print(f"  ch{i}: {peak_amp[i]*1e15:.0f} fT   pos {opm['pick'][i]}")
baseline_std = evoked[:, times < 0].std()
peak_val = np.abs(evoked[:, peak_window]).max()
print(f"peak/baseline ratio: {peak_val/baseline_std:.1f}  (want >>1)")
peak_window = (times >= 80) & (times <= 130)
peak_amp = np.abs(evoked[:, peak_window]).max(axis=1)
top = np.argsort(peak_amp)[::-1][:4]           # 4 strongest channels
print("top channels:", top, "positions:", opm["pick"][top])

plt.figure(figsize=(9,4))
for ch in top:
    plt.plot(times, evoked[ch]*1e15, label=f"ch{ch}")
plt.axvline(0,color="k",ls="--"); plt.axvline(100,color="r",ls=":")
plt.xlabel("ms from tone"); plt.ylabel("fT"); plt.legend(); plt.title("strongest channels only")
plt.savefig(os.path.join(FIG_DIR, "opm_auditory_strongest_channels.png"), dpi=120, bbox_inches="tight")
plt.show()