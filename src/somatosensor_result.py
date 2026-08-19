import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\Theod\Downloads\Megprojects\002"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUNS = [
    os.path.join(DATA_ROOT, "OPM", "Somatosensory", "run01.meg.mat"),
    os.path.join(DATA_ROOT, "OPM", "Somatosensory", "run02.meg.mat"),
]

SF = 2000


STIM_LAG_S = 0.0


BP_LO, BP_HI = 2, 100
NOTCH_HZ, NOTCH_Q = 60, 30

T_PRE, T_POST = 0.4, 0.3
BASE_START, BASE_END = -0.30, -0.05
RESP_LO, RESP_HI = 15, 60       
PEAK_SEARCH_LO, PEAK_SEARCH_HI = 0, 150
REJECT_PCTILE = 85
N_SURROGATE = 200
RNG = np.random.default_rng(0)

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)
RESP_MASK = (TIMES >= RESP_LO) & (TIMES <= RESP_HI)
SEARCH_MASK = (TIMES >= PEAK_SEARCH_LO) & (TIMES <= PEAK_SEARCH_HI)
POST_MASK = (TIMES >= 0) & (TIMES <= PEAK_SEARCH_HI)


def preprocess(x):
    bn, an = iirnotch(NOTCH_HZ, NOTCH_Q, SF)
    x = filtfilt(bn, an, x, axis=-1)
    b, a = butter(4, [BP_LO / (SF / 2), BP_HI / (SF / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def load_run(path, verbose=False):
    d = sio.loadmat(path)
    if verbose:
        keys = [(k, np.shape(v)) for k, v in d.items() if not k.startswith("__")]
        print(f"{os.path.basename(path)} fields: {keys}")
    data = preprocess(d["bexp"])
    trig = d["bexp_ext"].ravel()
    thr = (trig.max() + trig.min()) / 2
    above = trig > thr
    onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
    return data, onsets


def cut_epochs(data, onsets):
    onsets = onsets[(onsets - N_PRE >= 0) & (onsets + N_POST < data.shape[1])]
    if len(onsets) == 0:
        return np.empty((0, data.shape[0], N_PRE + N_POST))
    idx = onsets[:, None] + np.arange(-N_PRE, N_POST)[None, :]
    eps = data[:, idx].transpose(1, 0, 2)
    return eps - eps[:, :, BASE_MASK].mean(axis=2, keepdims=True)


def reject(eps, pctile=REJECT_PCTILE):
    ptp = np.ptp(eps, axis=2).max(axis=1)
    return eps[ptp < np.percentile(ptp, pctile)]


def gfp_of(eps):
    return np.sqrt((eps.mean(axis=0) ** 2).mean(axis=0))


runs = [load_run(p, verbose=(i == 0)) for i, p in enumerate(RUNS)]

for path, (_, onsets) in zip(RUNS, runs):
    isi = np.diff(onsets) / SF
    print(f"{os.path.basename(path)}: {len(onsets)} triggers, "
          f"median ISI {np.median(isi):.2f} s")

shift = int(STIM_LAG_S * SF)
epochs = np.concatenate([reject(cut_epochs(d, o + shift)) for d, o in runs])
print(f"\n{len(epochs)} trials after rejection")

gfp = gfp_of(epochs)
base_mean, base_sd = gfp[BASE_MASK].mean(), gfp[BASE_MASK].std()
peak_gfp = gfp[RESP_MASK].max()
peak_time = TIMES[RESP_MASK][np.argmax(gfp[RESP_MASK])]
search_peak_t = TIMES[SEARCH_MASK][np.argmax(gfp[SEARCH_MASK])]
print(f"peak GFP {peak_gfp * 1e15:.0f} fT at {peak_time:.0f} ms vs baseline "
      f"{base_mean * 1e15:.0f} +/- {base_sd * 1e15:.0f} fT")
print(f"largest peak in 0-{PEAK_SEARCH_HI} ms sits at {search_peak_t:.0f} ms")

n_needed = len(epochs)
null_peaks = np.empty(N_SURROGATE)
for i in range(N_SURROGATE):
    fake = []
    for data, onsets in runs:
        lo, hi = N_PRE, data.shape[1] - N_POST
        rand = RNG.integers(lo, hi, size=len(onsets))
        fake.append(reject(cut_epochs(data, rand)))
    fake = np.concatenate(fake)[:n_needed]
    null_peaks[i] = gfp_of(fake)[RESP_MASK].max()

p_emp = (np.sum(null_peaks >= peak_gfp) + 1) / (N_SURROGATE + 1)
print(f"surrogate null: median {np.median(null_peaks) * 1e15:.0f} fT, "
      f"95th pct {np.percentile(null_peaks, 95) * 1e15:.0f} fT, p = {p_emp:.4f}")


odd = epochs[::2].mean(axis=0)[:, POST_MASK].ravel()
even = epochs[1::2].mean(axis=0)[:, POST_MASK].ravel()
r = np.corrcoef(odd, even)[0, 1]
print(f"split-half evoked correlation, 0-{PEAK_SEARCH_HI} ms, all channels: r = {r:.2f}")

evoked = epochs.mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

axes[0].plot(TIMES, evoked.T * 1e15, lw=0.7, alpha=0.8)
axes[0].axvspan(RESP_LO, RESP_HI, color="0.9", zorder=0)
axes[0].axvline(0, color="k", ls=":", lw=1)
axes[0].set_ylabel("field (fT)")
axes[0].set_title(f"OPM somatosensory evoked, {len(epochs)} trials, all 30 channels")

axes[1].plot(TIMES, gfp * 1e15, color="k", lw=1.2, label="evoked GFP")
axes[1].axhline(np.percentile(null_peaks, 95) * 1e15, color="r", ls="--", lw=1,
                label="surrogate 95th pct")
axes[1].axvspan(RESP_LO, RESP_HI, color="0.9", zorder=0)
axes[1].axvline(0, color="k", ls=":", lw=1)
axes[1].set_xlabel("ms from stimulus")
axes[1].set_ylabel("GFP (fT)")
axes[1].legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "opm_somatosensory_gfp.png"), dpi=120, bbox_inches="tight")
plt.show()