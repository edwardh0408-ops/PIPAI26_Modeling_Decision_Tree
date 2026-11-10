"""Clinic no-show model, v1.

Split 400 appointments 70/30, stratified on no_show.
Fit one classification tree on the training appointments, score it
on the held-out appointments, and turn those scores into slot value.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

DATA_PATH = Path(__file__).with_name("clinic_appointments.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Prior misses and a text reminder dominate the no-show rate.
# Lead time has a real gradient on its own, so it is offered to the tree.
# Age, distance, new-patient status, and hour do not earn a stable split
# once those are available, and hour has no association with no-show.
FEATURES = ["prior_no_shows", "sms_reminder", "lead_time_days"]
POSITIVE = "no-show"
LABELS = ["checked-in", POSITIVE]

# Dollars per test appointment, from the clinic's cost table.
# TN hold + showed = 0. FP overbook + showed = -60.
# FN hold + missed = -120. TP overbook + missed = 0.
VALUE = {"tn": 0, "fp": -60, "fn": -120, "tp": 0}

# Upcoming appointments. The tree only reads prior misses, the reminder,
# and lead time; the other fields are kept so the record matches the brief.
UPCOMING = {
    "D": {
        "lead_time_days": 21,
        "prior_no_shows": 1,
        "age": 27,
        "distance_miles": 9.5,
        "sms_reminder": 0,
        "new_patient": 1,
        "appointment_hour": 15,
    },
    "E": {
        "lead_time_days": 3,
        "prior_no_shows": 0,
        "age": 61,
        "distance_miles": 2.0,
        "sms_reminder": 1,
        "new_patient": 0,
        "appointment_hour": 9,
    },
}

# Set in main() once v1 is fit. slot_decision reads this tree.
TREE = None


def scores(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp, fn, tp = matrix.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_no_show": precision_score(
            y_true, y_pred, pos_label=POSITIVE, zero_division=np.nan
        ),
        "recall_no_show": recall_score(y_true, y_pred, pos_label=POSITIVE),
    }


def format_scores(result):
    precision = result["precision_no_show"]
    precision_text = "undefined" if pd.isna(precision) else f"{precision:.3f}"
    return (
        f"TN {result['tn']}  FP {result['fp']}  FN {result['fn']}  TP {result['tp']}\n"
        f"accuracy {result['accuracy']:.3f}  "
        f"precision on no-shows {precision_text}  "
        f"recall on no-shows {result['recall_no_show']:.3f}"
    )


def slot_decision(appointment):
    """Overbook the slot if v1 predicts a no-show. Otherwise hold it."""
    if TREE is None:
        raise RuntimeError("Fit v1 before calling slot_decision.")
    row = pd.DataFrame([{feature: appointment[feature] for feature in FEATURES}])
    prediction = TREE.predict(row)[0]
    return "overbook" if prediction == POSITIVE else "hold"


def leaf_for(appointment):
    """Path from the root to the leaf, and the training rows in that leaf."""
    row = pd.DataFrame([{feature: appointment[feature] for feature in FEATURES}])
    node_id = int(TREE.apply(row)[0])
    nodes = TREE.tree_
    node = 0
    path = []
    while nodes.children_left[node] != nodes.children_right[node]:
        feature = FEATURES[nodes.feature[node]]
        threshold = float(nodes.threshold[node])
        value = appointment[feature]
        if value <= threshold:
            path.append(f"{feature} = {value:g}  <=  {threshold:.2f}")
            node = int(nodes.children_left[node])
        else:
            path.append(f"{feature} = {value:g}  >  {threshold:.2f}")
            node = int(nodes.children_right[node])
    # sklearn stores class shares in the leaf. Multiply by the row count
    # to recover how many training appointments landed there.
    shares = nodes.value[node_id][0]
    n_rows = int(nodes.n_node_samples[node_id])
    counts = {
        label: int(round(share * n_rows))
        for label, share in zip(TREE.classes_, shares)
    }
    return {"node": node_id, "path": path, "training_counts": counts}


def slot_value(result):
    """Total and per-appointment value from a confusion matrix."""
    total = sum(result[cell] * VALUE[cell] for cell in VALUE)
    n = result["tn"] + result["fp"] + result["fn"] + result["tp"]
    return total, total / n


def main():
    global TREE
    appointments = pd.read_csv(DATA_PATH)
    x = appointments[FEATURES]
    y = appointments["no_show"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Depth 3 is enough for two or three questions. A leaf must hold at
    # least 15 training appointments so a single row cannot become a rule.
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
    )
    tree.fit(x_train, y_train)
    TREE = tree

    print("SPLIT")
    print(f"rows {len(appointments)}  train {len(y_train)}  test {len(y_test)}")
    print(
        f"no-show rate  all { (y == POSITIVE).mean():.3%}  "
        f"train {(y_train == POSITIVE).mean():.3%} ({(y_train == POSITIVE).sum()})  "
        f"test {(y_test == POSITIVE).mean():.3%} ({(y_test == POSITIVE).sum()})"
    )
    print(f"random_state {RANDOM_STATE}  stratified on no_show")

    print("\nTREE")
    print(
        export_text(
            tree,
            feature_names=FEATURES,
            decimals=2,
            show_weights=True,
        )
    )
    print("class order in weights:", list(tree.classes_))
    print(
        "importances:",
        {
            name: round(float(weight), 3)
            for name, weight in zip(FEATURES, tree.feature_importances_)
        },
    )

    train_scores = scores(y_train, tree.predict(x_train))
    test_scores = scores(y_test, tree.predict(x_test))
    baseline_scores = scores(y_test, ["checked-in"] * len(y_test))

    print("\nTEST — baseline (predict show for every appointment)")
    print(format_scores(baseline_scores))
    print("\nTRAIN — v1")
    print(format_scores(train_scores))
    print("\nTEST — v1")
    print(format_scores(test_scores))

    print("\nDECISIONS")
    for name, appointment in UPCOMING.items():
        landed = leaf_for(appointment)
        counts = landed["training_counts"]
        print(f"{name}: {slot_decision(appointment)}")
        print(f"  leaf node {landed['node']}")
        for step in landed["path"]:
            print(f"  {step}")
        print(
            "  training rows in this leaf: "
            f"{counts['checked-in']} checked-in, {counts['no-show']} no-show"
        )

    print("\nVALUE on the 120 test appointments")
    for name, result in (("baseline", baseline_scores), ("v1", test_scores)):
        total, per_appointment = slot_value(result)
        print(
            f"{name}: "
            f"{result['tn']}×$0 + {result['fp']}×−$60 + "
            f"{result['fn']}×−$120 + {result['tp']}×$0 "
            f"= −${abs(total):,.0f} total, −${abs(per_appointment):,.2f} per appointment"
        )


if __name__ == "__main__":
    main()
