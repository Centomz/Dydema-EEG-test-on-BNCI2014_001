# Testing the Dydema EEG API on public motor-imagery data

A small, honest test of the [Dydema](https://dydema.com) EEG classification API ("Wizard") on a public motor-imagery dataset. No EEG hardware was used.

I am a 6th-year medical student building a biosignal engineering portfolio (PPG, ECG, EMG, and later EEG). This repository documents one concrete experiment: how well does a hosted classifier, built from a few labeled trials, work on EEG it has never seen?

> **Research use only.** This is an offline test on public data. It is not a medical device and makes no clinical claim. I am not affiliated with Dydema, and this is not an endorsement.

## Result

| | |
|---|---|
| Dataset | BCI Competition IV 2a (`BNCI2014_001` in MOABB), subject 1 |
| Task | Imagined left-hand vs right-hand movement |
| Trials | 144 train (session 1) / 144 test (session 2), balanced 72 / 72 per class |
| Signal | 22 EEG channels, 250 Hz, 0-4 s after the cue |
| Preprocessing | 8-32 Hz band-pass applied by MOABB (the API applies none), converted to microvolts |
| Validation accuracy (API `training_report`) | 0.861 balanced accuracy (chance 0.50), estimated within session 1 |
| **Held-out accuracy** | **0.910 (131 / 144), chance 0.50** |
| 95% interval (Wilson) | about 0.85 to 0.95 |
| Confusion | true left: 64 correct, 8 wrong; true right: 67 correct, 5 wrong |
| Mean confidence of predicted class | 0.81 |
| Cost | $0.43 in total (144 predictions at $0.003 each) |

**How the split works:** the model was created from session 1 only. Session 2 was recorded on a different day, its labels never left my machine, and the accuracy was computed locally from the API's predictions.

**Observation:** the held-out score (0.91) was slightly higher than the API's own validation estimate (0.86), which is the opposite of the optimism the API warns about. The validation estimate was computed inside session 1 with fewer trials, so I would not read much into the difference.

## Limits

- **One subject.** Subject 1 of this dataset is generally an easy one; other subjects would likely score lower. This says nothing about performance across people.
- One run, one split, no repeated trials or cross-subject test.
- Offline analysis of public data, not real-time use, not my own recordings.
- The selected method inside the API was not reported in the training report.
- Pre-filtering (8-32 Hz) was done by me through MOABB, not by the API.

## Reproduce

Requires Python 3.11 and a Dydema API key (never commit it).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DYDEMA_API_KEY="your_key"      # keep it in the environment only

python prepare_data.py                # downloads the dataset, writes subject1_train.npz / subject1_test.npz
python dydema_bci_test.py             # dry run, create, predict, report, optional delete
```

`dydema_bci_test.py` does a free dry run first and asks for confirmation before anything is charged. It sets the model to expire after 2 days and offers to delete it at the end, because hosting is billed monthly.

## Files

- `prepare_data.py`: loads the dataset through MOABB, keeps left vs right hand, cuts 0-4 s epochs, saves a train file (session 1) and a test file (session 2).
- `dydema_bci_test.py`: creates the task model from the train file, predicts on the test file, computes held-out accuracy and cost.
- `result.json`: the output of my run.

## References

- Dataset: Tangermann et al., "Review of the BCI Competition IV", Frontiers in Neuroscience, 2012.
- MOABB: Jayaram and Barachant, "MOABB: trustworthy algorithm benchmarking for BCIs", Journal of Neural Engineering, 2018.
- API documentation: https://developers.dydema.com
