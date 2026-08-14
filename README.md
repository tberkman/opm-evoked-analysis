# Auditory Evoked Response Extraction from Wearable OPM-MEG

Extraction and characterization of the auditory evoked response from
wearable optically-pumped-magnetometer (OPM) MEG recordings, using a
public OPM/SQUID/EEG dataset.

## What I did
- Trigger-locked averaging of ~340 auditory tones across two OPM runs
- Trigger onsets from the `bexp_ext` channel, corrected for a 20 ms acoustic lag
- Bandpass 2–20 Hz, per-channel baseline correction, rejection of the noisiest 15% of trials
- Statistical validation via global field power (GFP) at ~100 ms vs. pre-stimulus baseline

## Key result
The key result from this analysis is that auditory evoked response is actually present, albeit mild. After performing the averaging, the global field power peaked around 2.4 SD above the pre stimulus baseline, near 100 ms after the tone. This time locked response was obscured significantly by background power and oscillation, so it wasn't visible in the graphs clearly. 

## Noise characterization
After performing the power spectrum analysis, the data shows that it is dominated by 60 Hz line noise and harmonics. These were removed in our other calculations via a bandpass filter. The residual masking I found is low frequency, in band power at about 4-7 Hz. This overlapped directly with the evoked response and cannot be separated by frequency filtering. This is consistent with the signal extraction challenges described by others.


## Limitations
- SQUID comparison not possible: the `.con` files use a non-standard KIT
  channel type that MNE-Python cannot read, and no `.mat` export is provided.
  - A workaround is possible but not worth pursuing currently
- Single subject

## Files
- `src/evoked_response.py`
    - trigger extraction, epoching, averaging, GFP statistics
- `src/power_spectrum.py`
    - Welch PSD noise characterization
- `figures/` 
    - evoked butterfly plot, GFP plot, power spectra

## Data
Public OPM/SQUID/EEG dataset (subject s002). Data is not included in this repository.