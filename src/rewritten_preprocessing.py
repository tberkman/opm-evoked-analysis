import os
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, detrend

BASE = r"C:\Users\Theod\Downloads\Megprojects"

SUBJECTS = ["002", "005", "006", "093"]

CONDITIONS = {
    "raw":     set(),
    "hfc":     {"hfc"},
    "detrend": {"detrend"},
    "eog":     {"eog"},
    "all":     {"hfc", "detrend", "eog"},
}

BAD_CHANNELS = {"002": [], "005": [], "006": [], "093": []}

DETREND_SEG_S = 10.0
STG_TARGET = np.array([0.060, -0.020, 0.005])

SF = 2000
T_PRE, T_POST = 0.4, 0.3
BASE_START, BASE_END = -0.30, -0.05
LAG_S = 0.060
BP = (2, 20)
WIN = (70, 150)
REJECT_PCTILE = 85
N_SURROGATE = 500          # will rerun at 2000 later if necessary, low for now bc it takes forever to run at a high n

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)
WIN_MASK = (TIMES >= WIN[0]) & (TIMES <= WIN[1])

VERBOSE_ONCE = {"eog": True}


def hfc_projector(ori):

    n = ori / np.linalg.norm(ori, axis=1, keepdims=True)
    return np.eye(n.shape[0]) - n @ np.linalg.pinv(n)


def bandpass(x, lo, hi, sf):
    b, a = butter(4, [lo / (sf / 2), hi / (sf / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def cut(data, onsets, reject):
    onsets = onsets[(onsets - N_PRE >= 0) & (onsets + N_POST < data.shape[1])]
    if len(onsets) == 0:
        return np.empty((0, data.shape[0], N_PRE + N_POST)), onsets
    idx = onsets[:, None] + np.arange(-N_PRE, N_POST)[None, :]
    eps = data[:, idx].transpose(1, 0, 2)
    eps = eps - eps[:, :, BASE_MASK].mean(axis=2, keepdims=True)
    if not reject:
        return eps, onsets
    ptp = np.ptp(eps, axis=2).max(axis=1)
    keep = ptp < np.percentile(ptp, REJECT_PCTILE)
    return eps[keep], onsets[keep]


def load_eog(path_meg, n_meg_samples, verbose=False):

    path = path_meg.replace(".meg.mat", ".eeg.mat")
    if not os.path.exists(path):
        return None
    d = sio.loadmat(path)
    if "eeg_data" not in d:
        return None

    raw = np.asarray(d["eeg_data"], dtype=float)
    if raw.shape[0] > raw.shape[1]:
        raw = raw.T

    try:
        names = d["EEGinfo"]["ChannelName"][0, 0]
        n_named = len(names)
    except Exception:
        n_named = 2
    eog = raw[:n_named]

    try:
        sf_eeg = float(np.squeeze(d["EEGinfo"]["SampleFrequency"][0, 0]))
    except Exception:
        sf_eeg = 1000.0

    dur_eeg = raw.shape[1] / sf_eeg
    dur_meg = n_meg_samples / SF
    if abs(dur_eeg - dur_meg) > 1.0 / SF:
        print(f"    NOTE: EEG {dur_eeg:.3f} s vs MEG {dur_meg:.3f} s "
              f"({raw.shape[1]} @ {sf_eeg:.0f} Hz vs {n_meg_samples} @ {SF} Hz)")

    t_eeg = np.arange(raw.shape[1]) / sf_eeg
    t_meg = np.arange(n_meg_samples) / SF
    out = np.vstack([np.interp(t_meg, t_eeg, ch) for ch in eog])

    if verbose:
        dropped = raw.shape[0] - n_named
        print(f"    EOG: {raw.shape[0]} rows, {n_named} named, "
              f"{dropped} dropped as trigger/extra; {sf_eeg:.0f} Hz -> {SF} Hz")
    return out


def regress_eog(eps, eog_eps):

    n_tr, n_ch, n_t = eps.shape
    n_eog = eog_eps.shape[1]

    Xb = eog_eps[:, :, BASE_MASK].transpose(1, 0, 2).reshape(n_eog, -1).T
    Xb = np.hstack([Xb, np.ones((Xb.shape[0], 1))])
    Xf = eog_eps.transpose(1, 0, 2).reshape(n_eog, -1).T
    Xf = np.hstack([Xf, np.ones((Xf.shape[0], 1))])

    out = eps.copy()
    for c in range(n_ch):
        yb = eps[:, c, BASE_MASK].ravel()
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        out[:, c, :] = eps[:, c, :] - (Xf @ beta).reshape(n_tr, n_t)
    return out


def load_subject(subject, steps):
    paths = sorted(glob.glob(os.path.join(BASE, subject, "OPM", "Auditory",
                                          "run*.meg.mat")))
    if not paths:
        raise FileNotFoundError(f"{subject}: no auditory runs")
    bad = BAD_CHANNELS.get(subject, [])
    runs, eps_all, pos, ori_out, hfc_pairs = [], [], None, None, []

    for p in paths:
        d = sio.loadmat(p)
        raw = np.asarray(d["bexp"], dtype=float)
        good = np.setdiff1d(np.arange(raw.shape[0]), bad)
        raw = raw[good]
        ori = np.asarray(d["Qpick"])[good]
        if pos is None:
            pos = np.asarray(d["pick"])[good]
            ori_out = ori

        if "hfc" in steps:
            before = raw.copy()
            raw = hfc_projector(ori) @ raw
            hfc_pairs.append((before, raw))
        if "detrend" in steps:
            bps = np.arange(0, raw.shape[1], int(DETREND_SEG_S * SF))[1:]
            raw = detrend(raw, axis=-1, type="linear", bp=bps)

        data = bandpass(raw, BP[0], BP[1], SF)

        trig = d["bexp_ext"].ravel()
        above = trig > (trig.max() + trig.min()) / 2
        onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1 + int(LAG_S * SF)
        runs.append((data, onsets))

        eps, kept = cut(data, onsets, reject=True)
        if "eog" in steps:
            eog = load_eog(p, data.shape[1], verbose=VERBOSE_ONCE["eog"])
            VERBOSE_ONCE["eog"] = False
            if eog is None:
                print("    WARNING: EOG file not found or could not be loaded")
            else:
                eog_eps, _ = cut(bandpass(eog, BP[0], BP[1], SF), kept,
                                 reject=False)
                assert len(eog_eps) == len(eps), "EOG/MEG trial mismatch"
                eps = regress_eog(eps, eog_eps)
        eps_all.append(eps)

    return runs, np.concatenate(eps_all), pos, ori_out, hfc_pairs


def select_temporal(pos):
    sensor_pos = pos[::2]
    mask = np.zeros(pos.shape[0], dtype=bool)
    for sign in (-1, 1):
        target = STG_TARGET * np.array([sign, 1, 1])
        side = np.where(np.sign(sensor_pos[:, 0]) == sign)[0]
        if len(side):
            s = side[np.argmin(np.linalg.norm(sensor_pos[side] - target, axis=1))]
            mask[2 * s] = mask[2 * s + 1] = True
    return mask


def test(runs, eps, masks, seed):
    rng = np.random.default_rng(seed)
    nulls = {k: np.empty(N_SURROGATE) for k in masks}
    n_needed = len(eps)
    for i in range(N_SURROGATE):
        fake = []
        for data, onsets in runs:
            lo, hi = N_PRE, data.shape[1] - N_POST
            f, _ = cut(data, rng.integers(lo, hi, size=len(onsets)), reject=True)
            fake.append(f)
        avg = np.concatenate(fake)[:n_needed].mean(axis=0)
        for k, m in masks.items():
            nulls[k][i] = np.sqrt((avg[m] ** 2).mean(axis=0))[WIN_MASK].max()
    out = {}
    for k, m in masks.items():
        g = np.sqrt((eps.mean(axis=0)[m] ** 2).mean(axis=0))
        peak = g[WIN_MASK].max()
        out[k] = dict(peak=peak,
                      time=TIMES[WIN_MASK][np.argmax(g[WIN_MASK])],
                      p=(np.sum(nulls[k] >= peak) + 1) / (N_SURROGATE + 1))
    return out


rows = []
for i, subj in enumerate(SUBJECTS):
    print(f"\nsubject {subj}")
    for j, (name, steps) in enumerate(CONDITIONS.items()):
        try:
            runs, eps, pos, ori, hfc_pairs = load_subject(subj, steps)
        except Exception as e:
            print(f"  {name:<9} FAILED: {type(e).__name__}: {e}")
            continue
        masks = {"all": np.ones(pos.shape[0], dtype=bool),
                 "temporal": select_temporal(pos)}
        r = test(runs, eps, masks, seed=9000 + 53 * i + j)
        rows.append((subj, name, r))

        extra = ""
        if hfc_pairs:
            m = masks["temporal"]
            b = np.concatenate([v[0][m] for v in hfc_pairs], axis=1)
            a = np.concatenate([v[1][m] for v in hfc_pairs], axis=1)
            extra = (f"   HFC removed {100 * (1 - a.var() / b.var()):.0f}%"
                     f" of temporal-pair variance")
        print(f"  {name:<9} all {r['all']['peak']*1e15:6.0f} fT p={r['all']['p']:.4f}"
              f"   temporal {r['temporal']['peak']*1e15:6.0f} fT "
              f"@{r['temporal']['time']:4.0f} ms p={r['temporal']['p']:.4f}{extra}")

        # Fraction of the peak-latency topography's norm in the 3-D uniform-field
        # subspace; chance level for a random 30-vector is sqrt(3/30) = 32%.
        if name == "raw":
            g = np.sqrt((eps.mean(axis=0)[masks["temporal"]] ** 2).mean(axis=0))
            k = np.where(WIN_MASK)[0][np.argmax(g[WIN_MASK])]
            v = eps.mean(axis=0)[:, k]
            nrm = ori / np.linalg.norm(ori, axis=1, keepdims=True)
            frac = (np.linalg.norm(nrm @ np.linalg.pinv(nrm) @ v)
                    / np.linalg.norm(v))
            chance = np.sqrt(nrm.shape[1] / nrm.shape[0])
            print(f"            evoked topography at {TIMES[k]:.0f} ms sits "
                  f"{100*frac:.0f}% in the HFC uniform basis "
                  f"(chance {100*chance:.0f}%)")

print("\n" + "=" * 78)
print(f"{'subject':<9}{'condition':<11}{'all p':>10}{'temporal p':>14}"
      f"{'temporal peak':>16}{'latency':>10}")
print("=" * 78)
for subj, name, r in rows:
    print(f"{subj:<9}{name:<11}{r['all']['p']:>10.4f}{r['temporal']['p']:>14.4f}"
          f"{r['temporal']['peak']*1e15:>14.0f} fT{r['temporal']['time']:>8.0f} ms")
