"""
Somatosensory gate, run two ways.

The pre-registered gate said each subject's somatosensory control must clear
before its auditory result counts. Run with all 30 channels it failed for 005
(p = 0.36) and 093 (p = 0.67). That gate is structurally mismatched: it uses
whole-array GFP while the auditory test uses a 4-channel geometric selection,
and whole-array dilution is the very effect this project demonstrated.

This script reports BOTH versions for every subject. Neither is deleted.

Matched selector, frozen by anatomy: the 2 left-hemisphere sensors nearest the
hand knob at (-40, -25, +55) mm. Two sensors, four channels, matching the
auditory test's channel count. Right median nerve was stimulated, so the
response is contralateral and the selection is left-only.

BEFORE RUNNING, write your conclusion for each outcome here:
  matched gate clears in 005 and 093 ->
  matched gate still fails in one or both ->
"""

import os
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

BASE = r"C:\Users\Theod\Downloads\Megprojects"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SUBJECTS = ["002", "005", "006", "093"]
EXPLORATORY = ["002"]

HAND_KNOB = np.array([-40.0, -25.0, 55.0])
N_SELECT = 2

SF = 2000
T_PRE, T_POST = 0.4, 0.3
BASE_START, BASE_END = -0.30, -0.05
WIN = (15, 60)
REJECT_PCTILE = 85
N_SURROGATE = 2000

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)
WIN_MASK = (TIMES >= WIN[0]) & (TIMES <= WIN[1])


def preprocess(x):
    bn, an = iirnotch(60, 30, SF)
    x = filtfilt(bn, an, x, axis=-1)
    b, a = butter(4, [2 / (SF / 2), 100 / (SF / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def cut_epochs(data, onsets):
    onsets = onsets[(onsets - N_PRE >= 0) & (onsets + N_POST < data.shape[1])]
    if len(onsets) == 0:
        return np.empty((0, data.shape[0], N_PRE + N_POST))
    idx = onsets[:, None] + np.arange(-N_PRE, N_POST)[None, :]
    eps = data[:, idx].transpose(1, 0, 2)
    eps = eps - eps[:, :, BASE_MASK].mean(axis=2, keepdims=True)
    ptp = np.ptp(eps, axis=2).max(axis=1)
    return eps[ptp < np.percentile(ptp, REJECT_PCTILE)]


def load(subject):
    paths = sorted(glob.glob(os.path.join(BASE, subject, "OPM", "Somatosensory",
                                          "run*.meg.mat")))
    if not paths:
        raise FileNotFoundError(f"{subject}: no somatosensory runs")
    runs, eps, geo = [], [], None
    for p in paths:
        d = sio.loadmat(p)
        if geo is None:
            geo = d["pick"]
        data = preprocess(d["bexp"])
        trig = d["bexp_ext"].ravel()
        above = trig > (trig.max() + trig.min()) / 2
        onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
        runs.append((data, onsets))
        eps.append(cut_epochs(data, onsets))
    pos = geo * 1000.0 if np.abs(geo).max() < 1.0 else geo
    return runs, np.concatenate(eps), pos


def select_central(pos_ch):
    """FROZEN by anatomy. Left-hemisphere sensors nearest the hand knob."""
    sensor_pos = pos_ch[::2]
    left = np.where(sensor_pos[:, 0] < 0)[0]
    d = np.linalg.norm(sensor_pos[left] - HAND_KNOB, axis=1)
    chosen = left[np.argsort(d)[:N_SELECT]]
    mask = np.zeros(pos_ch.shape[0], dtype=bool)
    for s in chosen:
        mask[2 * s] = mask[2 * s + 1] = True
    return mask, np.sort(chosen), sensor_pos


def surrogate_nulls(runs, n_needed, masks, seed):
    rng = np.random.default_rng(seed)
    out = {k: np.empty(N_SURROGATE) for k in masks}
    for i in range(N_SURROGATE):
        fake = []
        for data, onsets in runs:
            lo, hi = N_PRE, data.shape[1] - N_POST
            fake.append(cut_epochs(data, rng.integers(lo, hi, size=len(onsets))))
        avg = np.concatenate(fake)[:n_needed].mean(axis=0)
        for k, m in masks.items():
            out[k][i] = np.sqrt((avg[m] ** 2).mean(axis=0))[WIN_MASK].max()
    return out


def score(eps, mask, nulls):
    g = np.sqrt((eps.mean(axis=0)[mask] ** 2).mean(axis=0))
    peak = g[WIN_MASK].max()
    t = TIMES[WIN_MASK][np.argmax(g[WIN_MASK])]
    p = (np.sum(nulls >= peak) + 1) / (N_SURROGATE + 1)
    return g, peak, t, p


results = {}
for i, subj in enumerate(SUBJECTS):
    try:
        runs, eps, pos = load(subj)
        mask_c, chosen, sensor_pos = select_central(pos)
        masks = {"all": np.ones(pos.shape[0], dtype=bool), "central": mask_c}
        nulls = surrogate_nulls(runs, len(eps), masks, seed=2000 + 31 * i)
        g_a, pk_a, t_a, p_a = score(eps, masks["all"], nulls["all"])
        g_c, pk_c, t_c, p_c = score(eps, mask_c, nulls["central"])
        results[subj] = dict(g_all=g_a, g_cen=g_c, p_all=p_a, p_cen=p_c,
                             t_all=t_a, t_cen=t_c, pk_all=pk_a, pk_cen=pk_c,
                             n=len(eps), chosen=chosen, sensor_pos=sensor_pos)
        tag = "EXPLORATORY" if subj in EXPLORATORY else "held out"
        print(f"\n=== subject {subj} ({tag}), {len(eps)} trials ===")
        for s in chosen:
            d = np.linalg.norm(sensor_pos[s] - HAND_KNOB)
            print(f"  selected s{s:<2d} ({sensor_pos[s,0]:6.0f}, {sensor_pos[s,1]:6.0f}, "
                  f"{sensor_pos[s,2]:6.0f})  {d:.0f} mm from hand knob")
        print(f"  gate as written (30 ch)  peak {pk_a*1e15:6.0f} fT @ {t_a:4.0f} ms   p = {p_a:.4f}")
        print(f"  matched gate     ( 4 ch)  peak {pk_c*1e15:6.0f} fT @ {t_c:4.0f} ms   p = {p_c:.4f}")
    except Exception as e:
        print(f"\n=== subject {subj} FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 66)
print(f"{'subj':<6}{'gate as written':>18}{'matched gate':>16}{'passes':>10}")
print("=" * 66)
for subj, r in results.items():
    both = ("both" if r["p_all"] < 0.05 and r["p_cen"] < 0.05
            else "matched only" if r["p_cen"] < 0.05
            else "as-written only" if r["p_all"] < 0.05 else "neither")
    print(f"{subj:<6}{r['p_all']:>18.4f}{r['p_cen']:>16.4f}{both:>10}")
print("\nReport both columns in the write-up. If the matched gate clears where the")
print("as-written gate did not, that is a finding about GFP dilution, not a repair.")

fig, axes = plt.subplots(len(results), 1, figsize=(9, 2.5 * len(results)), sharex=True)
axes = np.atleast_1d(axes)
for ax, (subj, r) in zip(axes, results.items()):
    ax.plot(TIMES, r["g_all"] * 1e15, color="0.6", lw=1.0, label="all 30 ch")
    ax.plot(TIMES, r["g_cen"] * 1e15, color="k", lw=1.4, label="central 4 ch")
    ax.axvspan(*WIN, color="0.9", zorder=0)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_title(f"subject {subj}   as-written p = {r['p_all']:.3f}   "
                 f"matched p = {r['p_cen']:.3f}", fontsize=10)
    ax.set_ylabel("GFP (fT)")
axes[0].legend(frameon=False, fontsize=8)
axes[-1].set_xlabel("ms from stimulus")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "opm_somatosensory_gate.png"), dpi=120,
            bbox_inches="tight")
plt.show()