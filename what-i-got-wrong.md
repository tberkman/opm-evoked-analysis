# What I got wrong
In the original version of this repository, there were 2 key mistakes present. 

First, the code was initially written to account for a 20 ms lag in the data collection, however, after reading the dataset's manual the true lag was 60 ms. This caused every epoch to be 40 ms early. The 20 ms assumption came from that being the value that would have been used in SQUID and EEG, as the dataset's manual was misunderstood.

The second mistake was an incorrect window analysis of the peaks in the data. Originally, the code would find the peak by searching through 80-130 ms and finding the highest point, before then seeing how far above the baseline mean that maximum was. In essence, if we just searched for the largest point, it would inherently be greater than our comparison, the mean, whether or not a response actually existed. To fix this, the code was changed to be comparing peaks against the maxima produced when the same recording is re-analysed with triggers at random times, rather than just against the mean.

Finally, after fixing both of these errors, the data showed auditory response across all 30 channels does not clear the null in any subject. 