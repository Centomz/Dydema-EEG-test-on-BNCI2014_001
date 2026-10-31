"""Step 3: load BCI Competition IV 2a (BNCI2014_001), subject 1, left vs right hand,
cut 0-4 s epochs after the cue, and save a train file (session 1) and a test file (session 2)."""
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

SUBJECT = 1

dataset = BNCI2014_001()
# MOABB epochs 0-4 s after the cue and, by default, band-pass filters 8-32 Hz
# (the Dydema API applies no band-pass itself, so we say so in the result note).
paradigm = LeftRightImagery(tmin=0.0, tmax=4.0, fmin=8.0, fmax=32.0)

epochs, y, meta = paradigm.get_data(dataset=dataset, subjects=[SUBJECT], return_epochs=True)
x = epochs.get_data() * 1e6  # MNE gives volts -> convert to microvolts
channels = list(epochs.ch_names)
sfreq = float(epochs.info["sfreq"])
y = np.asarray(y)
sessions = meta["session"].to_numpy()

print("Epochs array (trials, channels, samples):", x.shape)
print("Sampling rate:", sfreq, "Hz")
print("Channels:", channels)
print("Classes:", dict(zip(*np.unique(y, return_counts=True))))
print("Sessions:", dict(zip(*np.unique(sessions, return_counts=True))))
print("Signal std (uV), should be a few to tens:", round(float(x.std()), 2))

# Honest split: first session = train, second session = test.
first, second = sorted(np.unique(sessions))[:2]
for name, session in (("train", first), ("test", second)):
    mask = sessions == session
    np.savez(f"subject{SUBJECT}_{name}.npz",
             x=x[mask].astype("<f4"), y=y[mask], channels=np.array(channels), sfreq=sfreq)
    print(f"Saved subject{SUBJECT}_{name}.npz from session '{session}':", int(mask.sum()), "trials")