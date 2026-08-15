import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\Theod\Downloads\Megprojects\002"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

aud_run1 = os.path.join(DATA_ROOT, "OPM", "Auditory", "run01.meg.mat")

sf = 2000


def bandpass(x, lo, hi, sf, order=4):
    b, a = butter(order, [lo / (sf / 2), hi / (sf / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


opm = sio.loadmat(aud_run1)
data = bandpass(opm["bexp"], 1, 40, sf)
trig = opm["bexp_ext"].ravel()

thr = (trig.max() + trig.min()) / 2
above = trig > thr
onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
onsets = onsets + int(0.020 * sf)

tmin, tmax = -0.1, 0.4
n_pre, n_post = int(-tmin * sf), int(tmax * sf)
epochs = []
for o in onsets:
    if o + n_post < data.shape[1] and o - n_pre >= 0:
        epochs.append(data[:, o - n_pre:o + n_post])
epochs = np.array(epochs)

baseline = epochs[:, :, :n_pre].mean(axis=2, keepdims=True)
epochs = epochs - baseline
ptp = np.ptp(epochs, axis=2).max(axis=1)
good = ptp < np.percentile(ptp, 90)
print(f"keeping {good.sum()}/{len(good)} epochs")

evoked = epochs[good].mean(axis=0)
times = np.arange(-n_pre, n_post) / sf * 1000

peak_window = (times >= 80) & (times <= 130)
peak_amp = np.abs(evoked[:, peak_window]).max(axis=1)
top = np.argsort(peak_amp)[::-1][:4]
print("top channels:", top, "positions:", opm["pick"][top])

baseline_std = evoked[:, times < 0].std()
peak_val = np.abs(evoked[:, peak_window]).max()
print(f"peak/baseline ratio: {peak_val / baseline_std:.1f}")

plt.figure(figsize=(9, 4))
for ch in top:
    plt.plot(times, evoked[ch] * 1e15, label=f"ch{ch}")
plt.axvline(0, color="k", ls="--")
plt.axvline(100, color="r", ls=":")
plt.xlabel("ms from tone")
plt.ylabel("fT")
plt.legend()
plt.title("strongest channels")
plt.savefig(os.path.join(FIG_DIR, "opm_auditory_strongest_channels.png"), dpi=120, bbox_inches="tight")
plt.show()