"""
Confirmatory test of the geometry-selected auditory analysis.

The channel selector is FROZEN before this script sees subjects 005, 006, 093:
one sensor per hemisphere, whichever is nearest to superior temporal gyrus at
(+/-60, -20, +5) mm in the SPM right-handed frame. No free parameters, no
tuning, no sweep. Subject 002 generated the hypothesis and is reported
separately as exploratory - it is not part of the confirmatory result.
"""

import os
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import combine_pvalues
import matplotlib.pyplot as plt

BASE = r"C:\Users\Theod\Downloads\Megprojects"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EXPLORATORY = ["002"]
HELD_OUT = ["005", "006", "093"]

STG_TARGET = np.array([60.0, -20.0, 5.0])      # mirrored in x for the left side

SF = 2000
T_PRE, T_POST = 0.4, 0.3
BASE_START, BASE_END = -0.30, -0.05
REJECT_PCTILE = 85
N_SURROGATE = 2000

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)

TASKS = {
    "auditory": dict(folder="Auditory", lag=0.060, bp=(2, 20), notch=False,
                     win=(70, 150)),
    "somatosensory": dict(folder="Somatosensory", lag=0.0, bp=(2, 100), notch=True,
                          win=(15, 60)),
}
AUD_WIN = (TIMES >= TASKS["auditory"]["win"][0]) & (TIMES <= TASKS["auditory"]["win"][1])
SOM_WIN = (TIMES >= TASKS["somatosensory"]["win"][0]) & (TIMES <= TASKS["somatosensory"]["win"][1])
EARLY = (TIMES >= 15) & (TIMES <= 40)


def preprocess(x, bp, notch):
    if notch:
        bn, an = iirnotch(60, 30, SF)
        x = filtfilt(bn, an, x, axis=-1)
    b, a = butter(4, [bp[0] / (SF / 2), bp[1] / (SF / 2)], btype="band")
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


def load_task(subject, cfg):
    paths = sorted(glob.glob(os.path.join(BASE, subject, "OPM", cfg["folder"],
                                          "run*.meg.mat")))
    if not paths:
        raise FileNotFoundError(f"{subject}/{cfg['folder']}: no run*.meg.mat")
    runs, eps, geo = [], [], None
    for p in paths:
        d = sio.loadmat(p)
        if geo is None:
            geo = d["pick"]
        data = preprocess(d["bexp"], cfg["bp"], cfg["notch"])
        trig = d["bexp_ext"].ravel()
        above = trig > (trig.max() + trig.min()) / 2
        onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
        onsets = onsets + int(cfg["lag"] * SF)
        runs.append((data, onsets))
        eps.append(cut_epochs(data, onsets))
    pos = geo * 1000.0 if np.abs(geo).max() < 1.0 else geo
    return runs, np.concatenate(eps), pos, len(paths)


def select_temporal(pos_ch):
    """FROZEN. Nearest sensor per hemisphere to STG. Coordinates only."""
    sensor_pos = pos_ch[::2]
    mask = np.zeros(pos_ch.shape[0], dtype=bool)
    chosen = []
    for sign in (-1, 1):
        target = STG_TARGET * np.array([sign, 1, 1])
        side = np.where(np.sign(sensor_pos[:, 0]) == sign)[0]
        if len(side) == 0:
            continue
        s = side[np.argmin(np.linalg.norm(sensor_pos[side] - target, axis=1))]
        chosen.append(s)
        mask[2 * s] = mask[2 * s + 1] = True
    return mask, chosen, sensor_pos


def gfp(eps, mask):
    return np.sqrt((eps.mean(axis=0)[mask] ** 2).mean(axis=0))


def surrogate_nulls(runs, n_needed, masks, win_mask, seed):
    rng = np.random.default_rng(seed)
    out = {k: np.empty(N_SURROGATE) for k in masks}
    for i in range(N_SURROGATE):
        fake = []
        for data, onsets in runs:
            lo, hi = N_PRE, data.shape[1] - N_POST
            fake.append(cut_epochs(data, rng.integers(lo, hi, size=len(onsets))))
        avg = np.concatenate(fake)[:n_needed].mean(axis=0)
        for k, m in masks.items():
            out[k][i] = np.sqrt((avg[m] ** 2).mean(axis=0))[win_mask].max()
    return out


def test(eps, mask, nulls, win_mask):
    g = gfp(eps, mask)
    peak = g[win_mask].max()
    t = TIMES[win_mask][np.argmax(g[win_mask])]
    p = (np.sum(nulls >= peak) + 1) / (N_SURROGATE + 1)
    return dict(gfp=g, peak=peak, time=t, p=p,
                null95=np.percentile(nulls, 95))


def analyze(subject, seed):
    out = {"subject": subject}

    runs_s, eps_s, pos, nrun_s = load_task(subject, TASKS["somatosensory"])
    mask_t, chosen, sensor_pos = select_temporal(pos)
    all_mask = np.ones(pos.shape[0], dtype=bool)

    ev = eps_s.mean(axis=0)
    w = np.clip(np.abs(ev[:, EARLY]).max(axis=1) / ev[:, BASE_MASK].std(axis=1) - 1, 0, None)
    out["contra_x"] = np.average(pos[:, 0], weights=w) if w.sum() > 0 else np.nan

    nulls_s = surrogate_nulls(runs_s, len(eps_s), {"all": all_mask}, SOM_WIN, seed)
    out["som"] = test(eps_s, all_mask, nulls_s["all"], SOM_WIN)
    out["n_som"] = len(eps_s)

    runs_a, eps_a, _, nrun_a = load_task(subject, TASKS["auditory"])
    masks = {"all": all_mask, "temporal": mask_t}
    nulls_a = surrogate_nulls(runs_a, len(eps_a), masks, AUD_WIN, seed + 1)
    out["aud_all"] = test(eps_a, all_mask, nulls_a["all"], AUD_WIN)
    out["aud_tmp"] = test(eps_a, mask_t, nulls_a["temporal"], AUD_WIN)
    out["n_aud"] = len(eps_a)
    out["sensors"] = [(s, sensor_pos[s]) for s in chosen]
    out["runs"] = (nrun_s, nrun_a)
    return out


results = {}
for i, subj in enumerate(EXPLORATORY + HELD_OUT):
    try:
        results[subj] = analyze(subj, seed=1000 + 17 * i)
        r = results[subj]
        print(f"\n=== subject {subj} "
              f"({'EXPLORATORY' if subj in EXPLORATORY else 'held out'}) ===")
        print(f"  runs: {r['runs'][0]} somatosensory, {r['runs'][1]} auditory")
        for s, p3 in r["sensors"]:
            print(f"  selected sensor s{s:<2d} ({p3[0]:6.0f}, {p3[1]:6.0f}, {p3[2]:6.0f})")
        print(f"  frame check, contralateral mean x = {r['contra_x']:6.0f} mm "
              f"{'OK' if r['contra_x'] < 0 else '<-- NOT LEFT, do not trust this subject'}")
        print(f"  somatosensory   {r['n_som']:>4d} trials  peak {r['som']['peak']*1e15:6.0f} fT "
              f"@ {r['som']['time']:4.0f} ms   p = {r['som']['p']:.4f}")
        print(f"  auditory all    {r['n_aud']:>4d} trials  peak {r['aud_all']['peak']*1e15:6.0f} fT "
              f"@ {r['aud_all']['time']:4.0f} ms   p = {r['aud_all']['p']:.4f}")
        print(f"  auditory temporal          peak {r['aud_tmp']['peak']*1e15:6.0f} fT "
              f"@ {r['aud_tmp']['time']:4.0f} ms   p = {r['aud_tmp']['p']:.4f}")
    except Exception as e:
        print(f"\n=== subject {subj} FAILED: {type(e).__name__}: {e}")

valid = [s for s in HELD_OUT if s in results and results[s]["contra_x"] < 0]
print("\n" + "=" * 62)
print("CONFIRMATORY RESULT (held-out subjects only, frozen selector)")
print("=" * 62)
if len(valid) < len(HELD_OUT):
    dropped = [s for s in HELD_OUT if s not in valid]
    print(f"  excluded (frame check failed or load error): {dropped}")
if valid:
    ps = [results[s]["aud_tmp"]["p"] for s in valid]
    ts = [results[s]["aud_tmp"]["time"] for s in valid]
    for s, p, t in zip(valid, ps, ts):
        print(f"  {s}: p = {p:.4f}, peak at {t:.0f} ms")
    stat, p_comb = combine_pvalues(ps, method="fisher")
    print(f"\n  Fisher combined across {len(valid)} held-out subjects: p = {p_comb:.4f}")
    print(f"  peak latency spread: {min(ts):.0f}-{max(ts):.0f} ms "
          f"(an M100 should land near 90-110 ms in every subject)")
    print(f"\n  note: the surrogate p-value floor is {1/(N_SURROGATE+1):.4f}, "
          f"so Fisher cannot go below roughly {combine_pvalues([1/(N_SURROGATE+1)]*len(valid), method='fisher')[1]:.2e}")
else:
    print("  no usable held-out subjects")

fig, axes = plt.subplots(len(results), 1, figsize=(9, 2.6 * len(results)), sharex=True)
axes = np.atleast_1d(axes)
for ax, (subj, r) in zip(axes, results.items()):
    ax.plot(TIMES, r["aud_all"]["gfp"] * 1e15, color="0.6", lw=1.0, label="all 30 ch")
    ax.plot(TIMES, r["aud_tmp"]["gfp"] * 1e15, color="k", lw=1.4, label="temporal pair")
    ax.axhline(r["aud_tmp"]["null95"] * 1e15, color="r", ls="--", lw=1,
               label="surrogate 95th pct")
    ax.axvspan(*TASKS["auditory"]["win"], color="0.9", zorder=0)
    ax.axvline(0, color="k", ls=":", lw=1)
    tag = "exploratory" if subj in EXPLORATORY else "held out"
    ax.set_title(f"subject {subj} ({tag})  p = {r['aud_tmp']['p']:.3f}", fontsize=10)
    ax.set_ylabel("GFP (fT)")
axes[0].legend(frameon=False, fontsize=8)
axes[-1].set_xlabel("ms from sound onset")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "opm_auditory_multisubject.png"),
            dpi=120, bbox_inches="tight")
plt.show()