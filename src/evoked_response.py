import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\Theod\Downloads\Megprojects\002"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RUNS = [
    os.path.join(DATA_ROOT, "OPM", "Auditory", "run01.meg.mat"),
    os.path.join(DATA_ROOT, "OPM", "Auditory", "run02.meg.mat"),
]

SF = 2000

# OSE reference manual, OPM-MEG section: sound reached the ears 60 ms after
# trigger onset (20 ms ear tube + 40 ms soundboard). The 20 ms figure applies
# to the SQUID and EEG recordings only.
ACOUSTIC_LAG_S = 0.060

T_PRE, T_POST = 0.5, 0.5
BASE_START, BASE_END = -0.40, -0.05
M100_LO, M100_HI = 70, 150      # fixed a priori window, not chosen from the data
PEAK_SEARCH_LO, PEAK_SEARCH_HI = 0, 300
REJECT_PCTILE = 85
N_SURROGATE = 200
RNG = np.random.default_rng(0)

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)
M100_MASK = (TIMES >= M100_LO) & (TIMES <= M100_HI)
SEARCH_MASK = (TIMES >= PEAK_SEARCH_LO) & (TIMES <= PEAK_SEARCH_HI)


def bandpass(x, lo, hi, sf, order=4):
    b, a = butter(order, [lo / (sf / 2), hi / (sf / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def load_run(path):
    d = sio.loadmat(path)
    data = bandpass(d["bexp"], 2, 20, SF)
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


runs = [load_run(p) for p in RUNS]

for path, (_, onsets) in zip(RUNS, runs):
    isi = np.diff(onsets) / SF
    print(f"{os.path.basename(path)}: {len(onsets)} triggers, "
          f"median ISI {np.median(isi):.2f} s (expect ~1.7)")

# Side-by-side check of the two lag values. The 20 ms version is what the
# original script used
for lag in (0.020, 0.060):
    shift = int(lag * SF)
    eps = np.concatenate([reject(cut_epochs(d, o + shift)) for d, o in runs])
    g = gfp_of(eps)
    peak_t = TIMES[SEARCH_MASK][np.argmax(g[SEARCH_MASK])]
    print(f"lag {lag * 1000:.0f} ms -> {len(eps)} trials, "
          f"peak GFP {g[SEARCH_MASK].max() * 1e15:.0f} fT at {peak_t:.0f} ms")

shift = int(ACOUSTIC_LAG_S * SF)
epochs = np.concatenate([reject(cut_epochs(d, o + shift)) for d, o in runs])
print(f"\nprimary analysis: {len(epochs)} trials after rejection")

gfp = gfp_of(epochs)
base_mean, base_sd = gfp[BASE_MASK].mean(), gfp[BASE_MASK].std()
peak_gfp = gfp[M100_MASK].max()
peak_time = TIMES[M100_MASK][np.argmax(gfp[M100_MASK])]
sd_above = (peak_gfp - base_mean) / base_sd
print(f"peak GFP {peak_gfp * 1e15:.0f} fT at {peak_time:.0f} ms vs baseline "
      f"{base_mean * 1e15:.0f} +/- {base_sd * 1e15:.0f} fT ({sd_above:.1f} SD)")

n_needed = len(epochs)
null_peaks = np.empty(N_SURROGATE)
for i in range(N_SURROGATE):
    fake = []
    for data, onsets in runs:
        lo, hi = N_PRE, data.shape[1] - N_POST
        rand = RNG.integers(lo, hi, size=len(onsets))
        fake.append(reject(cut_epochs(data, rand)))
    fake = np.concatenate(fake)[:n_needed]
    null_peaks[i] = gfp_of(fake)[M100_MASK].max()

p_emp = (np.sum(null_peaks >= peak_gfp) + 1) / (N_SURROGATE + 1)
print(f"surrogate null: median {np.median(null_peaks) * 1e15:.0f} fT, "
      f"95th pct {np.percentile(null_peaks, 95) * 1e15:.0f} fT, p = {p_emp:.4f}")

# Split-half reliability: does the same waveform show up in independent halves?
odd, even = gfp_of(epochs[::2]), gfp_of(epochs[1::2])
r = np.corrcoef(odd[M100_MASK], even[M100_MASK])[0, 1]
print(f"split-half GFP correlation in {M100_LO}-{M100_HI} ms window: r = {r:.2f}")

plt.figure(figsize=(9, 4))
plt.plot(TIMES, gfp * 1e15, color="k", lw=1.2, label="evoked GFP")
plt.axhline(np.percentile(null_peaks, 95) * 1e15, color="r", ls="--", lw=1,
            label="surrogate 95th pct")
plt.axvspan(M100_LO, M100_HI, color="0.85", zorder=0)
plt.axvline(0, color="k", ls=":", lw=1)
plt.xlabel("ms from sound onset")
plt.ylabel("GFP (fT)")
plt.title(f"OPM auditory evoked GFP, {len(epochs)} trials, {ACOUSTIC_LAG_S * 1000:.0f} ms lag")
plt.legend(frameon=False)
plt.savefig(os.path.join(FIG_DIR, "opm_auditory_gfp.png"), dpi=120, bbox_inches="tight")
plt.show()