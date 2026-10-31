"""Local baseline (CSP + LDA) on the same split as the API test, plus a confusion-matrix figure.
Train on session 1, test on session 2. Nothing here is tuned on the test session."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

train = np.load("subject1_train.npz")
test = np.load("subject1_test.npz")
x_train, y_train = train["x"].astype("float64"), np.array([str(v) for v in train["y"]])
x_test, y_test = test["x"].astype("float64"), np.array([str(v) for v in test["y"]])
classes = ["left_hand", "right_hand"]

# Standard motor-imagery pipeline, settings fixed in advance (6 CSP components, log-variance, LDA).
# The data were already band-passed 8-32 Hz in prepare_data.py.
model = make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis())
model.fit(x_train, y_train)
pred = model.predict(x_test)

n = len(y_test)
k = int((pred == y_test).sum())
acc = k / n


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


lo, hi = wilson(k, n)
cm_base = confusion_matrix(y_test, pred, labels=classes)
print(f"CSP + LDA, trained on session 1, tested on session 2: {k}/{n} = {acc:.3f}  (95% CI {lo:.2f}-{hi:.2f})")
print("Confusion matrix (rows = true, columns = predicted), order:", classes)
print(cm_base)

# Dydema API result from dydema_bci_test.py (same train/test split), copied from its printed output.
cm_api = np.array([[64, 8], [5, 67]])
api_k = int(np.trace(cm_api))
api_lo, api_hi = wilson(api_k, int(cm_api.sum()))
print(f"Dydema API (from result.json run): {api_k}/{int(cm_api.sum())} = {api_k / cm_api.sum():.3f}  (95% CI {api_lo:.2f}-{api_hi:.2f})")

with open("result_baseline.json", "w") as f:
    json.dump({"method": "CSP(6 components, log) + LDA", "n_test": n, "correct": k,
               "accuracy": acc, "ci95_wilson": [lo, hi], "confusion_matrix": cm_base.tolist(),
               "class_order": classes}, f, indent=2)

# Figure: two confusion matrices, same split.
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, cm, title in ((axes[0], cm_api, f"Dydema API\n{api_k / cm_api.sum():.1%} ({api_k}/{int(cm_api.sum())})"),
                      (axes[1], cm_base, f"CSP + LDA (local)\n{acc:.1%} ({k}/{n})")):
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum() / 2)
    ax.set_xticks([0, 1], ["left", "right"])
    ax.set_yticks([0, 1], ["left", "right"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.sum() / 4 else "black", fontsize=14)
fig.suptitle("BNCI2014_001, subject 1: train session 1, test session 2 (chance = 50%)", fontsize=10)
fig.tight_layout()
fig.savefig("confusion_matrices.png", dpi=200)
print("Saved confusion_matrices.png and result_baseline.json")
