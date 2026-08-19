# opm-evoked-analysis

Analysis of the OPM-MEG recordings in [dataset, ATR/NICT - simultaneous OPM,
SQUID and EEG], covering the four subjects with OPM data.

Each subject had an array of 15 sensors concentrated over the superior and posterior scalp. Each of these sensors produces 2 different channels, 1 per axis. In this dataset, the auditory data M100 is not detectable in global field power across all of the 30 channels each subject has.  After restricting the data to be only the two sensors closest to the superior temporal gyrus, which contains the auditory cortex, it exceeds a permutation-based null built from the same recording in two of three held-out subjects with peaks at 96 and 109 ms. Averaging these sensors with the other 28 does not seem to let any notable response to auditory stimulus be seen. All-channel's probability value was 0.11-0.43, however, with just the two better positioned sensors it was 0.0010-0.0350, showing the difference in detectability. Furthermore, repeating this test on the somatosensory data, specified to the more somatosensory-positioned sensors reports results in the same direction, showing much lower p values with the better positioned sensors. The somatosensory data also reported two subjects clearing the null even with all channels averaged, unlike auditory which only cleared 2/3 times with the two auditory-specific sensors.

An earlier version of this repository reported a result that was wrong. See
[What I got wrong](#what-i-got-wrong).