import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\Theod\Downloads\Megprojects\002"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SF = 2000
T_PRE, T_POST = 0.4, 0.3
BASE_START, BASE_END = -0.30, -0.05
REJECT_PCTILE = 85
N_SURROGATE = 200
RNG = np.random.default_rng(0)

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)

TASKS = {
    "auditory": dict(folder="Auditory", lag=0.060, bp=(2, 20), notch=False,
                     win=(70, 150)),
    "somatosensory": dict(folder="Somatosensory", lag=0.0, bp=(2, 100), notch=True,
                          win=(15, 60)),
}


def preprocess(x, bp, notch):
    if notch:
        bn, an = iirnotch(60, 30, SF)
        x = filtfilt(bn, an, x, axis=-1)
    b, a = butter(4, [bp[0] / (SF / 2), bp[1] / (SF / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def cut_epochs(data, onsets):
    onsets = onsets[(onsets - N_PRE >= 0) & (onsets + N_POST < data.shape[1])]
    idx = onsets[:, None] + np.arange(-N_PRE, N_POST)[None, :]
    eps = data[:, idx].transpose(1, 0, 2)
    eps = eps - eps[:, :, BASE_MASK].mean(axis=2, keepdims=True)
    ptp = np.ptp(eps, axis=2).max(axis=1)
    return eps[ptp < np.percentile(ptp, REJECT_PCTILE)]


def load_task(cfg):
    runs, eps = [], []
    for run in ("run01.meg.mat", "run02.meg.mat"):
        d = sio.loadmat(os.path.join(DATA_ROOT, "OPM", cfg["folder"], run))
        data = preprocess(d["bexp"], cfg["bp"], cfg["notch"])
        trig = d["bexp_ext"].ravel()
        above = trig > (trig.max() + trig.min()) / 2
        onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
        runs.append((data, onsets + int(cfg["lag"] * SF)))
        eps.append(cut_epochs(data, onsets + int(cfg["lag"] * SF)))
    return runs, np.concatenate(eps)


geo = sio.loadmat(os.path.join(DATA_ROOT, "OPM", "Somatosensory", "run01.meg.mat"))
pos_ch = geo["pick"] * 1000.0 if np.abs(geo["pick"]).max() < 1.0 else geo["pick"]
sensor_pos = pos_ch[::2]                       # ch 2k and 2k+1 share a position
N_SENSOR = sensor_pos.shape[0]


def channels_of(sensor_idx):
    m = np.zeros(pos_ch.shape[0], dtype=bool)
    for s in sensor_idx:
        m[2 * s] = m[2 * s + 1] = True
    return m


def pick_temporal(k_per_hemi):
    """Sensors closest to the temporal scalp, chosen from coordinates ONLY.
    The M100 field maximum sits over lateral, inferior scalp, so score sensors
    by how lateral and how low they are. No data is consulted."""
    score = np.abs(sensor_pos[:, 0]) - sensor_pos[:, 2]
    left = np.where(sensor_pos[:, 0] < 0)[0]
    right = np.where(sensor_pos[:, 0] > 0)[0]
    sel = np.concatenate([
        left[np.argsort(score[left])[::-1][:k_per_hemi]],
        right[np.argsort(score[right])[::-1][:k_per_hemi]],
    ])
    return np.sort(sel)


def pick_central_left(k):
    """Control selector: sensors over left central scalp, where the contralateral
    hand response belongs. Also coordinates only."""
    score = -np.abs(sensor_pos[:, 0] + 40) + sensor_pos[:, 2]
    left = np.where(sensor_pos[:, 0] < 0)[0]
    return np.sort(left[np.argsort(score[left])[::-1][:k]])


def gfp(eps, mask):
    return np.sqrt((eps.mean(axis=0)[mask] ** 2).mean(axis=0))


def surrogate_nulls(runs, n_needed, masks, win_mask):
    """One surrogate loop, every channel subset scored from the same random
    epochs, so the comparisons are matched."""
    out = {name: np.empty(N_SURROGATE) for name in masks}
    for i in range(N_SURROGATE):
        fake = []
        for data, onsets in runs:
            lo, hi = N_PRE, data.shape[1] - N_POST
            fake.append(cut_epochs(data, RNG.integers(lo, hi, size=len(onsets))))
        fake = np.concatenate(fake)[:n_needed]
        avg = fake.mean(axis=0)
        for name, m in masks.items():
            out[name][i] = np.sqrt((avg[m] ** 2).mean(axis=0))[win_mask].max()
    return out


def report(name, eps, mask, nulls, win_mask):
    g = gfp(eps, mask)
    peak = g[win_mask].max()
    t = TIMES[win_mask][np.argmax(g[win_mask])]
    p = (np.sum(nulls >= peak) + 1) / (N_SURROGATE + 1)
    print(f"  {name:<28s} {mask.sum():>2d} ch   peak {peak*1e15:6.0f} fT @ {t:4.0f} ms   "
          f"null 95th {np.percentile(nulls, 95)*1e15:6.0f} fT   p = {p:.4f}")
    return g, peak, p


print("sensor positions (mm), one row per sensor:")
for s in range(N_SENSOR):
    print(f"  s{s:>2}  ({sensor_pos[s,0]:6.0f}, {sensor_pos[s,1]:6.0f}, {sensor_pos[s,2]:6.0f})")

temporal = pick_temporal(2)
central = pick_central_left(3)
print(f"\ntemporal sensors (a priori, 2 per hemisphere): {temporal.tolist()}")
for s in temporal:
    print(f"   s{s}  ({sensor_pos[s,0]:.0f}, {sensor_pos[s,1]:.0f}, {sensor_pos[s,2]:.0f})")
print(f"left central sensors (control selector): {central.tolist()}")

# --- corrected contralaterality check, early window only -------------------
cfg = TASKS["somatosensory"]
runs_s, eps_s = load_task(cfg)
early = (TIMES >= 15) & (TIMES <= 40)
ev = eps_s.mean(axis=0)
snr_early = np.abs(ev[:, early]).max(axis=1) / ev[:, BASE_MASK].std(axis=1)
w = np.clip(snr_early - 1, 0, None)
print(f"\ncontralaterality check, 15-40 ms, weights = max(SNR-1, 0):")
print(f"  weighted mean x = {np.average(pos_ch[:, 0], weights=w):.0f} mm "
      f"(negative = left = contralateral to the stimulated right hand)")

# --- primary test: auditory, geometry-selected channels --------------------
cfg = TASKS["auditory"]
runs_a, eps_a = load_task(cfg)
win_a = (TIMES >= cfg["win"][0]) & (TIMES <= cfg["win"][1])

masks_a = {
    "all channels": np.ones(pos_ch.shape[0], dtype=bool),
    "temporal (a priori)": channels_of(temporal),
}
for k in (1, 3, 4, 5):
    masks_a[f"temporal k={k}/hemi"] = channels_of(pick_temporal(k))

nulls_a = surrogate_nulls(runs_a, len(eps_a), masks_a, win_a)

print(f"\nauditory, {len(eps_a)} trials, window {cfg['win'][0]}-{cfg['win'][1]} ms")
print("  PRIMARY (channel set fixed in advance from coordinates):")
g_all, _, _ = report("all channels", eps_a, masks_a["all channels"], nulls_a["all channels"], win_a)
g_tmp, _, p_tmp = report("temporal (a priori)", eps_a, masks_a["temporal (a priori)"],
                         nulls_a["temporal (a priori)"], win_a)
print("  EXPLORATORY sweep - these p-values are uncorrected, do not quote the best one:")
for k in (1, 3, 4, 5):
    key = f"temporal k={k}/hemi"
    report(key, eps_a, masks_a[key], nulls_a[key], win_a)

# --- control: does the selector machinery work where a response exists? ----
cfg = TASKS["somatosensory"]
win_s = (TIMES >= cfg["win"][0]) & (TIMES <= cfg["win"][1])
masks_s = {
    "all channels": np.ones(pos_ch.shape[0], dtype=bool),
    "left central (a priori)": channels_of(central),
}
nulls_s = surrogate_nulls(runs_s, len(eps_s), masks_s, win_s)
print(f"\nsomatosensory control, {len(eps_s)} trials, window {cfg['win'][0]}-{cfg['win'][1]} ms")
for name in masks_s:
    report(name, eps_s, masks_s[name], nulls_s[name], win_s)

# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(TIMES, g_all * 1e15, color="0.55", lw=1.1, label="all 30 channels")
axes[0].plot(TIMES, g_tmp * 1e15, color="k", lw=1.4, label="temporal sensors only")
axes[0].axhline(np.percentile(nulls_a["temporal (a priori)"], 95) * 1e15,
                color="r", ls="--", lw=1, label="surrogate 95th pct (temporal)")
axes[0].axvspan(*cfg["win"] if False else TASKS["auditory"]["win"], color="0.9", zorder=0)
axes[0].axvline(0, color="k", ls=":", lw=1)
axes[0].set_xlabel("ms from sound onset")
axes[0].set_ylabel("GFP (fT)")
axes[0].set_title(f"auditory GFP by channel set (p = {p_tmp:.3f} temporal)")
axes[0].legend(frameon=False, fontsize=8)

sel = np.zeros(N_SENSOR, dtype=bool)
sel[temporal] = True
axes[1].scatter(sensor_pos[~sel, 0], sensor_pos[~sel, 2], s=80, c="0.75",
                edgecolor="k", linewidth=0.4, label="other sensors")
axes[1].scatter(sensor_pos[sel, 0], sensor_pos[sel, 2], s=110, c="crimson",
                edgecolor="k", linewidth=0.5, label="temporal selection")
axes[1].set_xlabel("x (R+, mm)")
axes[1].set_ylabel("z (S+, mm)")
axes[1].set_aspect("equal")
axes[1].set_title("array seen from the front")
axes[1].legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "opm_auditory_temporal_channels.png"),
            dpi=120, bbox_inches="tight")
plt.show()