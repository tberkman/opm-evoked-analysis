import os
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, detrend

BASE = r"C:\Users\Theod\Downloads\Megprojects"

SUBJECTS = ["002", "005", "006", "093"]

BAD_CHANNELS = {"002": [], "005": [], "006": [], "093": []}

HAND_KNOB = np.array([-0.040, -0.025, 0.055])   # metres, left hemisphere
N_SELECT = 2                                     # sensors -> 4 channels

SF = 2000
T_PRE, T_POST = 0.4, 0.3
BASE_START, BASE_END = -0.30, -0.05
LAG_S = 0.0
BP = (2, 100)
NOTCH_HZ, NOTCH_Q = 60, 30
WIN = (15, 120)
REJECT_PCTILE = 85

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)
WIN_MASK = (TIMES >= WIN[0]) & (TIMES <= WIN[1])


def preprocess(x):
    bn, an = iirnotch(NOTCH_HZ, NOTCH_Q, SF)
    x = filtfilt(bn, an, x, axis=-1)
    b, a = butter(4, [BP[0] / (SF / 2), BP[1] / (SF / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def cut(data, onsets):
    onsets = onsets[(onsets - N_PRE >= 0) & (onsets + N_POST < data.shape[1])]
    idx = onsets[:, None] + np.arange(-N_PRE, N_POST)[None, :]
    eps = data[:, idx].transpose(1, 0, 2)
    eps = eps - eps[:, :, BASE_MASK].mean(axis=2, keepdims=True)
    ptp = np.ptp(eps, axis=2).max(axis=1)
    return eps[ptp < np.percentile(ptp, REJECT_PCTILE)]


def select_central(pos):
    sensor_pos = pos[::2]
    left = np.where(sensor_pos[:, 0] < 0)[0]
    d = np.linalg.norm(sensor_pos[left] - HAND_KNOB, axis=1)
    chosen = left[np.argsort(d)[:N_SELECT]]
    mask = np.zeros(pos.shape[0], dtype=bool)
    for s in chosen:
        mask[2 * s] = mask[2 * s + 1] = True
    return mask, np.sort(chosen), sensor_pos


def load_subject(subject, hfc=False):
    paths = sorted(glob.glob(os.path.join(BASE, subject, "OPM", "Somatosensory",
                                          "run*.meg.mat")))
    if not paths:
        raise FileNotFoundError(f"{subject}: no somatosensory runs")
    bad = BAD_CHANNELS.get(subject, [])
    eps_all, pos, ori_out = [], None, None

    for p in paths:
        d = sio.loadmat(p)
        raw = np.asarray(d["bexp"], dtype=float)
        good = np.setdiff1d(np.arange(raw.shape[0]), bad)
        raw = raw[good]
        ori = np.asarray(d["Qpick"])[good]
        if pos is None:
            pos = np.asarray(d["pick"])[good]
            ori_out = ori

        if hfc:
            n = ori / np.linalg.norm(ori, axis=1, keepdims=True)
            raw = (np.eye(n.shape[0]) - n @ np.linalg.pinv(n)) @ raw

        data = preprocess(raw)
        trig = d["bexp_ext"].ravel()
        above = trig > (trig.max() + trig.min()) / 2
        onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1 + int(LAG_S * SF)
        eps_all.append(cut(data, onsets))

    return np.concatenate(eps_all), pos, ori_out


def uniform_fraction(v, ori):
    n = ori / np.linalg.norm(ori, axis=1, keepdims=True)
    return (np.linalg.norm(n @ np.linalg.pinv(n) @ v) / np.linalg.norm(v),
            np.sqrt(n.shape[1] / n.shape[0]))


rows = []
for subj in SUBJECTS:
    try:
        eps, pos, ori = load_subject(subj, hfc=False)
        eps_hfc, _, _ = load_subject(subj, hfc=True)
    except Exception as e:
        print(f"{subj}: FAILED {type(e).__name__}: {e}")
        continue

    mask, chosen, sensor_pos = select_central(pos)
    ev = eps.mean(axis=0)

    g = np.sqrt((ev[mask] ** 2).mean(axis=0))
    k = np.where(WIN_MASK)[0][np.argmax(g[WIN_MASK])]
    frac, chance = uniform_fraction(ev[:, k], ori)

    peak_raw = g[WIN_MASK].max()
    g_hfc = np.sqrt((eps_hfc.mean(axis=0)[mask] ** 2).mean(axis=0))
    peak_hfc = g_hfc[WIN_MASK].max()

    edge = "  <-- peak at window edge; consider widening WIN" if k in (
        np.where(WIN_MASK)[0][0], np.where(WIN_MASK)[0][-1]) else ""

    print(f"\nsubject {subj} | {len(eps)} trials")
    for s in chosen:
        dist = np.linalg.norm(sensor_pos[s] - HAND_KNOB) * 1000
        print(f"  selected s{s:<2d} ({sensor_pos[s,0]*1000:6.0f},"
              f" {sensor_pos[s,1]*1000:6.0f}, {sensor_pos[s,2]*1000:6.0f}) mm"
              f"   {dist:.0f} mm from hand knob")
    print(f"  peak {peak_raw*1e15:6.0f} fT @ {TIMES[k]:4.0f} ms{edge}")
    print(f"  after HFC {peak_hfc*1e15:6.0f} fT "
          f"({100*(1 - peak_hfc/peak_raw):+.0f}% change)")
    print(f"  topography at {TIMES[k]:.0f} ms sits {100*frac:.0f}% in the "
          f"HFC uniform basis (chance {100*chance:.0f}%)")

    rows.append((subj, TIMES[k], peak_raw, peak_hfc, frac, chance))

if rows:
    print("\n" + "=" * 74)
    print(f"{'subject':<9}{'latency':>9}{'peak':>10}{'after HFC':>12}"
          f"{'change':>9}{'uniform %':>12}")
    print("=" * 74)
    for subj, t, pr, ph, frac, chance in rows:
        print(f"{subj:<9}{t:>7.0f} ms{pr*1e15:>8.0f} fT{ph*1e15:>10.0f} fT"
              f"{100*(1-ph/pr):>+8.0f}%{100*frac:>11.0f}%")
    print(f"\nchance level {100*rows[0][5]:.0f}%")
    print("Auditory reference values: 70 / 86 / 76 / 81%.")
    print("Comparable somatosensory projection would implicate array geometry")
