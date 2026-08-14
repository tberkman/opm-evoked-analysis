
import scipy.io as sio, numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import os
DATA_ROOT = r"C:\Users\Theod\Downloads\Megprojects\002"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)   
aud_run1 = os.path.join(DATA_ROOT, "OPM", "Auditory", "run01.meg.mat")
aud_run2 = os.path.join(DATA_ROOT, "OPM", "Auditory", "run02.meg.mat")
rest_run1 = os.path.join(DATA_ROOT, "OPM", "Rest", "run01.meg.mat")
sf = 2000
def psd(path):
    d = sio.loadmat(path)
    f, p = welch(d["bexp"], fs=sf, nperseg=8192, axis=-1)  # finer resolution
    return f, p.mean(axis=0)

f_aud, p_aud = psd(aud_run1)
f_rest, p_rest = psd(rest_run1)

plt.figure(figsize=(11,5))
plt.semilogy(f_aud, p_aud, label="auditory", lw=1.2)
plt.semilogy(f_rest, p_rest, label="rest", lw=1.2, alpha=0.75)
plt.xlim(0, 20)                              # ZOOM to your passband
plt.axvspan(8, 12, color="green", alpha=0.1, label="alpha band (8-12 Hz)")
plt.xlabel("frequency (Hz)"); plt.ylabel("power (log)")
plt.title("20 hz passband analysis")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
plt.savefig(os.path.join(FIG_DIR, "opm_spectrum20hz.png"), dpi=120, bbox_inches="tight")
band = (f_aud > 2) & (f_aud < 20)
print("strongest in 2-20 Hz:", np.round(f_aud[band][np.argsort(p_aud[band])[::-1][:5]], 1))