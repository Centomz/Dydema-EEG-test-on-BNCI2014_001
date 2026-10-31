"""Steps 6-8: train a Dydema task model on session 1, test on session 2 (never sent
with the create request), report held-out accuracy and cost, then optionally delete."""
import os
import json
import uuid
import base64
import numpy as np
import requests

API = "https://dydema--eegapi-gateway.modal.run/v1"
HEADERS = {"Authorization": f"Bearer {os.environ['DYDEMA_API_KEY']}"}
ID_FILE = "task_model_id_bci.txt"
DESCRIPTION = "Motor imagery: left vs right hand, 0-4 s after cue. Labels: left_hand, right_hand."
assert len(DESCRIPTION) <= 100, len(DESCRIPTION)

total_cost = 0.0


def post(url, payload, timeout, key=None):
    """POST with an Idempotency-Key, stop with a readable error, track cost."""
    global total_cost
    headers = {**HEADERS, "Idempotency-Key": key or str(uuid.uuid4())}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not r.ok:
        e = r.json()
        raise SystemExit(f"{r.status_code} {e['code']}: {e['message']} {e['alternatives']}")
    total_cost += float(r.headers.get("X-Usage-Usd", 0) or 0)
    return r.json()


def encode(x):
    x = np.ascontiguousarray(x, dtype="<f4")
    return {"x": base64.b64encode(x.tobytes()).decode("ascii"), "shape": list(x.shape)}


train = np.load("subject1_train.npz")
test = np.load("subject1_test.npz")
channels = [str(c) for c in train["channels"]]
sfreq = int(float(train["sfreq"]))
y_train = [str(v) for v in train["y"]]
y_test = [str(v) for v in test["y"]]
signal_info = {"sampling_rate_hz": sfreq, "units": "uV", "channels": channels}
print(f"Train: {train['x'].shape}, test: {test['x'].shape}, {sfreq} Hz, {len(channels)} channels")

# 1) Create (free dry run first). Only TRAIN trials and labels are sent.
create_body = {**encode(train["x"]), **signal_info, "labels": y_train, "description": DESCRIPTION,
               "name": "BNCI2014_001 subject 1 left vs right", "expires_in_days": 2}
print("\nDry run:")
print(json.dumps(post(f"{API}/task-models", {**create_body, "dry_run": True}, 600), indent=2))
if input("Create the task model for real? [y/N] ").strip().lower() != "y":
    raise SystemExit("Stopped after the dry run.")

task = post(f"{API}/task-models", create_body, 900)
with open(ID_FILE, "w") as f:  # save the id right away
    f.write(task["id"])
print("\nTask model:", task["id"], task["status"])
print("Training report (validation score is optimistic):")
print(json.dumps(task["training_report"], indent=2))

# 2) Predict on the held-out session (labels stay on this machine).
pred = post(f"{API}/task-models/{task['id']}/predictions", {**encode(test["x"]), **signal_info}, 600)
data = pred["data"]
predicted = [d["label"] for d in data]
classes = sorted(set(y_test))
correct = np.array([p == t for p, t in zip(predicted, y_test)])
recalls = [np.mean([p == c for p, t in zip(predicted, y_test) if t == c]) for c in classes]
confidence = np.mean([max(d["probabilities"].values()) for d in data])

print("\n=== HELD-OUT RESULT (session 2, never seen by the model) ===")
print(f"Trials: {len(y_test)}")
print(f"Accuracy: {correct.mean():.3f}  (chance = {1 / len(classes):.2f})")
print(f"Balanced accuracy: {np.mean(recalls):.3f}")
for c in classes:
    print(f"  true {c}: " + ", ".join(
        f"predicted {k} = {sum(1 for p, t in zip(predicted, y_test) if t == c and p == k)}" for k in classes))
print(f"Mean confidence of the predicted class: {confidence:.3f}")
print(f"Total cost so far (X-Usage-Usd headers): ${total_cost:.3f}")

with open("result.json", "w") as f:
    json.dump({"task_model_id": task["id"], "n_train": len(y_train), "n_test": len(y_test),
               "training_report": task["training_report"], "heldout_accuracy": float(correct.mean()),
               "heldout_balanced_accuracy": float(np.mean(recalls)), "mean_confidence": float(confidence),
               "cost_usd_headers": total_cost}, f, indent=2)
print("Saved result.json")

# 3) Delete to stop hosting charges.
if input("\nDelete the task model now? [y/N] ").strip().lower() == "y":
    r = requests.delete(f"{API}/task-models/{task['id']}", headers=HEADERS, timeout=60)
    print("Delete:", r.status_code)
else:
    print(f"Kept. It expires automatically after 2 days. Id is in {ID_FILE}.")