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

def power_spectrum(path, label):
    d = sio.loadmat(path)
    data = d["bexp"]                   
    freqs, psd = welch(data, fs=sf, nperseg=4096, axis=-1)
    psd_mean = psd.mean(axis=0)         
    return freqs, psd_mean

f_aud, p_aud = power_spectrum(aud_run1, "auditory")
f_rest, p_rest = power_spectrum(rest_run1, "rest")

plt.figure(figsize=(11,5))
plt.semilogy(f_aud, p_aud, label="auditory", lw=1)
plt.semilogy(f_rest, p_rest, label="rest", lw=1, alpha=0.7)
plt.xlim(0, 100)                         # zoom to 0-100 Hz where the action is
plt.axvline(50, color="gray", ls=":", alpha=0.5, label="50 Hz (JP line)")
plt.axvline(60, color="orange", ls=":", alpha=0.5, label="60 Hz")
plt.xlabel("frequency (Hz)"); plt.ylabel("power (log scale)")
plt.title("OPM power spectrum")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "opm_spectrum.png"), dpi=120, bbox_inches="tight")

plt.show()

# also print where the biggest peaks are
band = (f_aud > 5) & (f_aud < 100)
top_freqs = f_aud[band][np.argsort(p_aud[band])[::-1][:5]]
print("strongest frequencies (Hz):", np.round(top_freqs, 1))