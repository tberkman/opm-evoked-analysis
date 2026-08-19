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

N_PRE, N_POST = int(T_PRE * SF), int(T_POST * SF)
TIMES = (np.arange(N_PRE + N_POST) - N_PRE) / SF * 1000
BASE_MASK = (TIMES >= BASE_START * 1000) & (TIMES <= BASE_END * 1000)

TASKS = {
    "somatosensory": dict(
        folder="Somatosensory", lag=0.0, bp=(2, 100), notch=True, win=(15, 150)),
    "auditory": dict(
        folder="Auditory", lag=0.060, bp=(2, 20), notch=False, win=(70, 150)),
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


def channel_snr(cfg):
    """Per-channel peak response in the window, in units of that channel's own
    baseline noise. Descriptive - used only to locate the array, not to test."""
    eps = []
    for run in ("run01.meg.mat", "run02.meg.mat"):
        d = sio.loadmat(os.path.join(DATA_ROOT, "OPM", cfg["folder"], run))
        data = preprocess(d["bexp"], cfg["bp"], cfg["notch"])
        trig = d["bexp_ext"].ravel()
        above = trig > (trig.max() + trig.min()) / 2
        onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
        eps.append(cut_epochs(data, onsets + int(cfg["lag"] * SF)))
    evoked = np.concatenate(eps).mean(axis=0)
    win = (TIMES >= cfg["win"][0]) & (TIMES <= cfg["win"][1])
    amp = np.abs(evoked[:, win]).max(axis=1)
    noise = evoked[:, BASE_MASK].std(axis=1)
    return amp / noise, amp


geo = sio.loadmat(os.path.join(DATA_ROOT, "OPM", "Somatosensory", "run01.meg.mat"))
pos, ori = geo["pick"], geo["Qpick"]
print("CoordType:", geo["CoordType"])
print("PositionFile:", geo["PositionFile"])

if np.abs(pos).max() < 1.0:
    pos = pos * 1000.0
    print("positions look like metres, converted to mm")

print(f"\n{pos.shape[0]} channels, "
      f"{len(np.unique(np.round(pos, 3), axis=0))} unique positions "
      f"(expect 15 if each sensor contributes 2 axes)")
print("extent (mm):  x %.0f..%.0f   y %.0f..%.0f   z %.0f..%.0f"
      % (pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max(),
         pos[:, 2].min(), pos[:, 2].max()))
print("centroid (mm):", np.round(pos.mean(axis=0), 1))
print("orientation row norms:", np.round(np.linalg.norm(ori, axis=1), 3)[:5], "...")

snr, amp = {}, {}
for name, cfg in TASKS.items():
    snr[name], amp[name] = channel_snr(cfg)
    order = np.argsort(snr[name])[::-1][:5]
    print(f"\ntop 5 channels, {name} (window {cfg['win'][0]}-{cfg['win'][1]} ms):")
    for c in order:
        print(f"  ch{c:>2}  SNR {snr[name][c]:5.1f}  amp {amp[name][c]*1e15:6.0f} fT  "
              f"pos ({pos[c,0]:6.0f}, {pos[c,1]:6.0f}, {pos[c,2]:6.0f})  "
              f"ori ({ori[c,0]:5.2f}, {ori[c,1]:5.2f}, {ori[c,2]:5.2f})")

# Right median nerve was stimulated, so the somatosensory response should sit in
# the LEFT hemisphere. In a RAS frame that means negative x. If it does, the
# coordinate convention is confirmed physiologically.
w = snr["somatosensory"]
print(f"\nSNR-weighted mean x of somatosensory response: "
      f"{np.average(pos[:, 0], weights=w):.0f} mm "
      f"(negative = left hemisphere in RAS)")

VIEWS = [(0, 1, "x (R+)", "y (A+)", "axial, from above"),
         (0, 2, "x (R+)", "z (S+)", "coronal, from front"),
         (1, 2, "y (A+)", "z (S+)", "sagittal, from right")]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for row, name in enumerate(TASKS):
    for col, (i, j, xl, yl, title) in enumerate(VIEWS):
        ax = axes[row, col]
        s = ax.scatter(pos[:, i], pos[:, j], c=snr[name], s=90,
                       cmap="viridis", edgecolor="k", linewidth=0.4)
        ax.quiver(pos[:, i], pos[:, j], ori[:, i], ori[:, j],
                  color="0.4", width=0.004, scale=18)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
        if col == 0:
            ax.set_title(f"{name}\n{title}")
        else:
            ax.set_title(title)
        plt.colorbar(s, ax=ax, label="peak / baseline SD")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "opm_sensor_geometry.png"), dpi=120, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=snr["somatosensory"],
           s=90, cmap="viridis", edgecolor="k", linewidth=0.4)
ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2],
          ori[:, 0], ori[:, 1], ori[:, 2], length=20, color="0.4")
ax.set_xlabel("x (R+)")
ax.set_ylabel("y (A+)")
ax.set_zlabel("z (S+)")
ax.set_title("OPM array, coloured by somatosensory SNR")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "opm_sensor_geometry_3d.png"), dpi=120, bbox_inches="tight")
plt.show()