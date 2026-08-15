import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy import stats
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\Theod\Downloads\Megprojects\002"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

aud_run1 = os.path.join(DATA_ROOT, "OPM", "Auditory", "run01.meg.mat")
aud_run2 = os.path.join(DATA_ROOT, "OPM", "Auditory", "run02.meg.mat")

sf = 2000


def bandpass(x, lo, hi, sf, order=4):
    b, a = butter(order, [lo / (sf / 2), hi / (sf / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def get_epochs(path):
    d = sio.loadmat(path)
    data = bandpass(d["bexp"], 2, 20, sf)
    trig = d["bexp_ext"].ravel()
    thr = (trig.max() + trig.min()) / 2
    above = trig > thr
    onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
    onsets = onsets + int(0.020 * sf)
    n_pre, n_post = int(0.1 * sf), int(0.4 * sf)
    eps = []
    for o in onsets:
        if o - n_pre >= 0 and o + n_post < data.shape[1]:
            seg = data[:, o - n_pre:o + n_post]
            seg = seg - seg[:, :n_pre].mean(axis=1, keepdims=True)
            eps.append(seg)
    return np.array(eps)


ep1 = get_epochs(aud_run1)
ep2 = get_epochs(aud_run2)
epochs = np.concatenate([ep1, ep2], axis=0)
print(f"run01: {len(ep1)}  run02: {len(ep2)}  total: {len(epochs)} tones")

ptp = np.ptp(epochs, axis=2).max(axis=1)
good = ptp < np.percentile(ptp, 85)
epochs = epochs[good]
print(f"after rejection: {len(epochs)} tones")

n_pre = int(0.1 * sf)
evoked = epochs.mean(axis=0)
times = (np.arange(evoked.shape[1]) - n_pre) / sf * 1000

gfp = np.sqrt((evoked ** 2).mean(axis=0))
baseline_gfp = gfp[times < 0]
peak_gfp = gfp[(times >= 80) & (times <= 130)].max()
sd_above = (peak_gfp - baseline_gfp.mean()) / baseline_gfp.std()
print(f"peak GFP {peak_gfp * 1e15:.0f} fT vs baseline "
      f"{baseline_gfp.mean() * 1e15:.0f} ± {baseline_gfp.std() * 1e15:.0f} fT")
print(f"peak is {sd_above:.1f} SD above baseline")

post_rms = np.sqrt((epochs[:, :, (times >= 80) & (times <= 130)] ** 2).mean(axis=(1, 2)))
pre_rms = np.sqrt((epochs[:, :, times < 0] ** 2).mean(axis=(1, 2)))
t, p = stats.ttest_rel(post_rms, pre_rms)
print(f"RMS post vs pre: t={t:.2f}, p={p:.4f}")

plt.figure(figsize=(9, 4))
plt.plot(times, gfp * 1e15)
plt.axvline(0, color="k", ls="--")
plt.axvline(100, color="r", ls=":")
plt.xlabel("ms from tone")
plt.ylabel("GFP (fT)")
plt.title("global field power")
plt.savefig(os.path.join(FIG_DIR, "opm_auditory_gfp.png"), dpi=120, bbox_inches="tight")
plt.show()