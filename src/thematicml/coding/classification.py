"""Optional supervised coding trained on researcher-labelled records.

The classifier works at record level (labels belong to records). Train/test
splits are grouped by respondent so that no respondent appears on both sides.
For evaluation the TF-IDF vectorizer is re-fitted on the training records of
each split, so no vocabulary or IDF statistics come from the test records.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thematicml.coding.extraction import ASSIGN_COLUMNS, CODE_COLUMNS
from thematicml.features import FeatureSet


def train_record_classifier(
    records: pd.DataFrame,
    segments: pd.DataFrame,
    features: FeatureSet,
    random_state: int = 42,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """records: record_id, record_key, text, label. Returns codes, assignments, evaluation."""
    # imported on first use: slow to load in the browser edition
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    labelled = records[records["label"].notna()].copy()
    info: dict = {"available": False}
    if labelled.empty:
        info["reason"] = "No labelled records (label column empty or not mapped)."
        return pd.DataFrame(columns=CODE_COLUMNS), pd.DataFrame(columns=ASSIGN_COLUMNS), info
    counts = labelled["label"].value_counts()
    if len(counts) < 2:
        info["reason"] = "At least two distinct labels are required to train a classifier."
        return pd.DataFrame(columns=CODE_COLUMNS), pd.DataFrame(columns=ASSIGN_COLUMNS), info

    X_all = features.vectorizer.transform(records["text"].tolist())
    rec_index = {rid: i for i, rid in enumerate(records["record_id"])}
    li = np.array([rec_index[r] for r in labelled["record_id"]])
    y = labelled["label"].to_numpy()
    groups = labelled["record_key"].to_numpy()

    evaluation: dict = {"available": True, "n_labelled": int(len(labelled)), "class_distribution": counts.to_dict()}
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    lab_text = labelled["text"].to_numpy()

    def fit_predict(tr, te):
        """Vectorizer and classifier fitted on the training records only."""
        vec = clone(features.vectorizer)
        model = clone(clf).fit(vec.fit_transform(lab_text[tr]), y[tr])
        return model.predict(vec.transform(lab_text[te]))

    if len(labelled) >= 20 and len(set(groups)) >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, te = next(gss.split(li, y, groups))
        if len(set(y[tr])) >= 2:
            pred = fit_predict(tr, te)
            classes = sorted(set(y))
            p, r, f, s = precision_recall_fscore_support(y[te], pred, labels=classes, zero_division=0)
            majority = pd.Series(y[tr]).value_counts().idxmax()
            evaluation.update(
                {
                    "split": {"train": int(len(tr)), "test": int(len(te)), "grouped_by": "record_key (respondent)",
                              "vectorizer": "re-fitted on training records only"},
                    "accuracy": round(float(accuracy_score(y[te], pred)), 4),
                    "majority_baseline_accuracy": round(float(np.mean(y[te] == majority)), 4),
                    "macro_precision": round(float(p.mean()), 4),
                    "macro_recall": round(float(r.mean()), 4),
                    "macro_f1": round(float(f.mean()), 4),
                    "per_class": [
                        {"label": c, "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                         "f1": round(float(f[i]), 4), "support": int(s[i])}
                        for i, c in enumerate(classes)
                    ],
                    "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y[te], pred, labels=classes).tolist()},
                }
            )
        k = min(5, len(set(groups)))
        if k >= 3:
            accs, f1s = [], []
            for ftr, fte in GroupKFold(n_splits=k).split(lab_text, y, groups):
                if len(set(y[ftr])) < 2:
                    continue
                fp = fit_predict(ftr, fte)
                accs.append(accuracy_score(y[fte], fp))
                f1s.append(f1_score(y[fte], fp, average="macro", zero_division=0))
            if accs:
                evaluation["cross_validation"] = {
                    "folds": len(accs), "grouped_by": "record_key (respondent)",
                    "accuracy_mean": round(float(np.mean(accs)), 4), "accuracy_sd": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0,
                    "macro_f1_mean": round(float(np.mean(f1s)), 4), "macro_f1_sd": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0,
                    "fold_accuracy": [round(float(a), 4) for a in accs],
                }
    else:
        evaluation["note"] = "Too few labelled records/respondents (need >= 20 records and >= 4 respondents) for a held-out evaluation; no metrics reported."

    clf.fit(X_all[li], y)
    unl = records[records["label"].isna()]
    pred_lookup: dict = {}
    if len(unl):
        ui = np.array([rec_index[r] for r in unl["record_id"]])
        proba = clf.predict_proba(X_all[ui])
        best = proba.argmax(axis=1)
        for rid, b, pr in zip(unl["record_id"], best, proba):
            pred_lookup[rid] = (clf.classes_[b], float(pr[b]))
    truth = dict(zip(labelled["record_id"], labelled["label"]))

    codes = [
        {"code_key": f"sup:{c}", "label": str(c), "description": f"Records labelled or classified as “{c}”.",
         "method": "supervised", "feature": "logistic regression on TF-IDF", "category": None}
        for c in clf.classes_
    ]
    assigns = []
    for pos, rid in enumerate(segments["record_id"].tolist()):
        if rid in truth:
            assigns.append((f"sup:{truth[rid]}", pos, "researcher-label", 1.0, None, None, None, False))
        elif rid in pred_lookup:
            c, pr = pred_lookup[rid]
            assigns.append((f"sup:{c}", pos, "supervised", round(pr, 4), None, None, None, False))
    evaluation["predicted_records"] = len(pred_lookup)
    evaluation["note_assignment"] = "Record-level labels/predictions are attached to every segment of the record."
    return pd.DataFrame(codes, columns=CODE_COLUMNS), pd.DataFrame(assigns, columns=ASSIGN_COLUMNS), evaluation
