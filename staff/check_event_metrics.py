#!/usr/bin/env python3
"""Course-staff check: what the HMM smoother does for fall alerts, and where the evaluation leaks.

Rebuilds the authors' random forest + HMM smoother (notebooks/model_fitting.ipynb, cells
108 to 111) from the committed data file and scores it the way a fall alarm would be
judged, by falls caught and false alarms per hour, not only frame by frame. It prints:

1. Random forest alone against random forest + Viterbi smoothing, under the authors'
   leave-one-sequence-out protocol and under leave-one-subject-out, with frame-level
   accuracy, macro F1, fall precision and recall, falls caught and false alarms per hour.
2. The same with every feature that depends on absolute x/y room position replaced by
   features that do not (see LOCATION-FREE FEATURES below), and where in the room the
   falls happen: the chest tag's mean x/y position during the first, second and third
   fall of each recording, across the 25 recordings.
3. The length of the activity chunks the authors' HMM and HSMM classify, by class, and
   the Poisson duration log-probability that the HSMM gives a fall-length chunk under
   each class. This is the duration leak described in the staff README.

DEFINITIONS
  frame          One row of data/ConfLongDemo_JSI.txt. Each row is one tag's position
                 reading; the other three tags carry their last reading forward, as in
                 the authors' pipeline. Rows are about 27 ms apart (median).
  fall event     A maximal run of consecutive frames labelled "falling" in one sequence.
                 The data has 75 (three in each of the 25 sequences).
  fall segment   A maximal run of consecutive frames that a model predicts as "falling".
  caught         A fall event is caught if at least one predicted fall segment overlaps
                 the event widened by --tolerance frames on each side (default 40 frames,
                 about one second).
  false alarm    A predicted fall segment that overlaps no widened fall event.
  per hour       False alarms divided by the recorded time of the held-out sequences,
                 summed over all folds: 91.0 minutes for all 25 sequences.

THE MODEL (as in the notebook)
  The 32 features of cell 108 (12 tag coordinates, the norm of each tag's position vector,
  its frame-to-frame change, its rolling mean and standard deviation over 10 frames, and
  four cross-tag summaries), standardized on the training folds; a random forest with
  balanced class weights and min_samples_leaf=2 on six merged activities; Viterbi
  decoding with the RF class probabilities as emission scores, a uniform start and a
  frame-level transition matrix counted from the training folds with add-one smoothing.
  Cell numbers count every cell of the notebook from the top, starting at 1.
  The notebook uses 200 trees; this script uses 50 by default to stay fast, which moves
  the frame-level numbers by about 0.01.

LOCATION-FREE FEATURES
  Every feature is unchanged if a whole recording is shifted horizontally: each tag's
  height z, its frame-to-frame change and its rolling standard deviation over 10 frames;
  each tag's horizontal speed sqrt(dx^2 + dy^2) and its rolling mean over 10 frames; the
  chest and belt heights relative to the mean ankle height; and the horizontal distance
  from the chest to the midpoint of the ankles. (The norm of a position vector, which the
  authors call acceleration magnitude, depends on where the person stands, so it is out.)

Needs numpy, pandas and scikit-learn. About 70 seconds on an 18-core laptop with the
default 50 trees, about 17 CPU-minutes in all, so several minutes on four cores;
leave-one-sequence-out fits 25 forests per feature set and dominates the time.

Usage:  python staff/check_event_metrics.py [--data PATH] [--trees N] [--tolerance FRAMES]
"""
import argparse
import math
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

TAGS = {
    "010-000-024-033": "ANKLE_LEFT",
    "010-000-030-096": "ANKLE_RIGHT",
    "020-000-033-111": "CHEST",
    "020-000-032-221": "BELT",
}
SENSORS = ["ANKLE_LEFT", "ANKLE_RIGHT", "BELT", "CHEST"]
# the authors' merge of the 11 recorded activities into 6 (models/preprocessing.py)
MERGE = {
    "walking": "walking",
    "falling": "falling",
    "lying": "lying",
    "lying down": "lying",
    "sitting": "sitting",
    "sitting down": "sitting",
    "sitting on the ground": "sitting",
    "standing up from lying": "standing",
    "standing up from sitting": "standing",
    "standing up from sitting on the ground": "standing",
    "on all fours": "on_all_fours",
}
W = 10  # rolling window in frames, as in the notebook


def load(path):
    df = pd.read_csv(path, header=None, names=["seq", "tag", "timestamp", "date", "x", "y", "z", "activity"])
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S:%f")
    df = df.sort_values(["seq", "date"], kind="stable").reset_index(drop=True)
    df["sensor"] = df["tag"].map(TAGS)
    df["label"] = df["activity"].map(MERGE)
    raw = [f"{s}_{a}" for s in SENSORS for a in "xyz"]
    for s in SENSORS:
        mine = df["sensor"] == s
        for a in "xyz":
            df[f"{s}_{a}"] = np.where(mine, df[a], np.nan)
    df[raw] = df.groupby("seq")[raw].ffill()
    df[raw] = df.groupby("seq")[raw].bfill()
    return df, raw


def roll(df, col, how):
    g = df.groupby("seq")[col]
    if how == "mean":
        return g.transform(lambda v: v.rolling(W, min_periods=1).mean())
    return g.transform(lambda v: v.rolling(W, min_periods=1).std().fillna(0))


def authors_features(df, raw):
    """The 32 features of notebook cell 108."""
    for s in SENSORS:
        df[f"{s}_mag"] = np.sqrt(sum(df[f"{s}_{a}"] ** 2 for a in "xyz"))
        df[f"{s}_jerk"] = df.groupby("seq")[f"{s}_mag"].diff().fillna(0)
    mags = [f"{s}_mag" for s in SENSORS]
    df["mag_mean"] = df[mags].mean(axis=1)
    df["mag_std"] = df[mags].std(axis=1)
    df["mag_max"] = df[mags].max(axis=1)
    df["mag_range"] = df[mags].max(axis=1) - df[mags].min(axis=1)
    for s in SENSORS:
        df[f"{s}_roll_mean"] = roll(df, f"{s}_mag", "mean")
        df[f"{s}_roll_std"] = roll(df, f"{s}_mag", "std")
    return (
        raw
        + mags
        + [f"{s}_jerk" for s in SENSORS]
        + [f"{s}_roll_mean" for s in SENSORS]
        + [f"{s}_roll_std" for s in SENSORS]
        + ["mag_mean", "mag_std", "mag_max", "mag_range"]
    )


def location_free_features(df):
    """Features that do not change if a recording is shifted in x and y."""
    cols = []
    for s in SENSORS:
        df[f"{s}_dz"] = df.groupby("seq")[f"{s}_z"].diff().fillna(0)
        df[f"{s}_z_rstd"] = roll(df, f"{s}_z", "std")
        dx = df.groupby("seq")[f"{s}_x"].diff().fillna(0)
        dy = df.groupby("seq")[f"{s}_y"].diff().fillna(0)
        df[f"{s}_hspeed"] = np.sqrt(dx**2 + dy**2)
        df[f"{s}_hspeed_rmean"] = roll(df, f"{s}_hspeed", "mean")
        cols += [f"{s}_z", f"{s}_dz", f"{s}_z_rstd", f"{s}_hspeed", f"{s}_hspeed_rmean"]
    ankle_z = 0.5 * (df["ANKLE_LEFT_z"] + df["ANKLE_RIGHT_z"])
    df["chest_above_ankles"] = df["CHEST_z"] - ankle_z
    df["belt_above_ankles"] = df["BELT_z"] - ankle_z
    mx = 0.5 * (df["ANKLE_LEFT_x"] + df["ANKLE_RIGHT_x"])
    my = 0.5 * (df["ANKLE_LEFT_y"] + df["ANKLE_RIGHT_y"])
    df["chest_ankle_hdist"] = np.sqrt((df["CHEST_x"] - mx) ** 2 + (df["CHEST_y"] - my) ** 2)
    return cols + ["chest_above_ankles", "belt_above_ankles", "chest_ankle_hdist"]


def runs(mask):
    """(start, end) index pairs, inclusive, of the maximal runs of True in a boolean array."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def viterbi(post, A):
    """The notebook's hmm_smooth (cell 110), vectorized over classes."""
    C = A.shape[0]
    logA, logE = np.log(A + 1e-10), np.log(post + 1e-10)
    v = logE[0] - np.log(C)
    back = np.zeros(post.shape, dtype=int)
    for t in range(1, len(post)):
        cand = v[:, None] + logA  # (from, to)
        back[t] = cand.argmax(0)
        v = cand.max(0) + logE[t]
    path = np.empty(len(post), dtype=int)
    path[-1] = v.argmax()
    for t in range(len(post) - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def events(pred, truth, fall, tol):
    """Falls caught, number of fall events and false alarms in one sequence."""
    true_ev = runs(truth == fall)
    pred_ev = runs(pred == fall)

    def overlaps(p, e):
        return not (p[1] < e[0] - tol or p[0] > e[1] + tol)

    caught = sum(any(overlaps(p, e) for p in pred_ev) for e in true_ev)
    false = sum(not any(overlaps(p, e) for e in true_ev) for p in pred_ev)
    return caught, len(true_ev), false


def evaluate(df, feats, folds, seqs, Y, le, hours, trees, tol, title):
    t0 = time.time()
    C, fall = len(le.classes_), int(np.flatnonzero(le.classes_ == "falling")[0])
    pred = {"RF": {}, "RF + Viterbi": {}}
    for test in folds:
        train = [s for s in seqs if s not in test]
        tr = df["seq"].isin(train).to_numpy()
        scaler = StandardScaler().fit(df.loc[tr, feats].to_numpy())
        A = np.ones((C, C))
        for s in train:
            np.add.at(A, (Y[s][:-1], Y[s][1:]), 1)
        A /= A.sum(axis=1, keepdims=True)
        rf = RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42
        )
        rf.fit(scaler.transform(df.loc[tr, feats].to_numpy()), le.transform(df.loc[tr, "label"]))
        for s in test:
            post = rf.predict_proba(scaler.transform(df.loc[df["seq"] == s, feats].to_numpy()))
            pred["RF"][s] = post.argmax(1)
            pred["RF + Viterbi"][s] = viterbi(post, A)
    print(f"\n{title}  ({len(folds)} folds, {time.time() - t0:.0f} s)")
    print(f"  {'model':<14}{'acc':>6}{'macroF1':>9}{'fall P':>8}{'fall R':>8}{'caught':>9}{'false alarms':>14}{'per hour':>10}")
    yt = np.concatenate([Y[s] for s in seqs])
    for name, P in pred.items():
        yp = np.concatenate([P[s] for s in seqs])
        tp = np.sum((yp == fall) & (yt == fall))
        prec = tp / max(np.sum(yp == fall), 1)
        rec = tp / np.sum(yt == fall)
        caught = total = false = 0
        for s in seqs:
            c, n, f = events(P[s], Y[s], fall, tol)
            caught, total, false = caught + c, total + n, false + f
        print(
            f"  {name:<14}{accuracy_score(yt, yp):6.3f}{f1_score(yt, yp, average='macro'):9.3f}"
            f"{prec:8.2f}{rec:8.2f}{caught:>5d}/{total:<3d}{false:>14d}{false / hours:10.1f}"
        )


def fall_places(df):
    """Chest-tag position during each fall: (fall number within its recording, mean x, mean y)."""
    chest = df[df["sensor"] == "CHEST"]
    out = []
    for _, g in chest.groupby("seq"):
        for i, (a, b) in enumerate(runs((g["label"] == "falling").to_numpy())):
            out.append((i + 1, g["x"].to_numpy()[a : b + 1].mean(), g["y"].to_numpy()[a : b + 1].mean()))
    return pd.DataFrame(out, columns=["fall", "x", "y"]), chest


def chunk_lengths(df, seqs):
    lengths = {}
    for s in seqs:
        lab = df.loc[df["seq"] == s, "label"].to_numpy()
        cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(lab)]):
            lengths.setdefault(lab[a], []).append(b - a)
    return lengths


def log_poisson(k, lam):
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--data", default=str(repo / "data" / "ConfLongDemo_JSI.txt"))
    ap.add_argument("--trees", type=int, default=50, help="trees per random forest (default 50; the notebook used 200)")
    ap.add_argument("--tolerance", type=int, default=40, help="frames either side of a fall that still count as a hit (default 40)")
    args = ap.parse_args()

    df, raw = load(args.data)
    f32 = authors_features(df, raw)
    free = location_free_features(df)
    seqs = sorted(df["seq"].unique())
    le = LabelEncoder().fit(df["label"])
    Y = {s: le.transform(df.loc[df["seq"] == s, "label"]) for s in seqs}
    hours = df.groupby("seq")["date"].agg(lambda d: (d.max() - d.min()).total_seconds()).sum() / 3600
    fall_runs = [e for s in seqs for e in runs(Y[s] == int(np.flatnonzero(le.classes_ == "falling")[0]))]
    print(f"{len(df):,} frames in {len(seqs)} sequences, {hours * 60:.1f} minutes recorded")
    secs = []
    for s in seqs:
        t = df.loc[df["seq"] == s, "date"].to_numpy()
        for a, b in runs(Y[s] == int(np.flatnonzero(le.classes_ == "falling")[0])):
            secs.append((t[b] - t[a]) / np.timedelta64(1, "s"))
    print(f"{len(fall_runs)} fall events lasting {np.mean(secs):.2f} s on average (median {np.median(secs):.2f} s);"
          f" {np.mean(df['label'] == 'falling'):.1%} of frames are labelled falling")
    print(f"{args.trees} trees per forest; an event is caught within {args.tolerance} frames")
    walk = df[df["activity"] == "walking"]
    print("median height z of each tag's own readings while walking (metres): "
          + ", ".join(f"{s.lower()} {walk.loc[walk['sensor'] == s, 'z'].median():.2f}" for s in SENSORS))

    by_sequence = [[s] for s in seqs]
    by_subject = [[s for s in seqs if s[0] == p] for p in sorted({s[0] for s in seqs})]
    common = (seqs, Y, le, hours, args.trees, args.tolerance)
    print("\n1. The authors' 32 features")
    evaluate(df, f32, by_sequence, *common, "leave-one-sequence-out (the authors' protocol)")
    evaluate(df, f32, by_subject, *common, "leave-one-subject-out (all five recordings of a person held out together)")
    print(f"\n2. Without absolute x/y room position ({len(free)} location-free features)")
    evaluate(df, free, by_sequence, *common, "leave-one-sequence-out")
    evaluate(df, free, by_subject, *common, "leave-one-subject-out")
    places, chest = fall_places(df)
    print("\n  Where the falls happen (chest tag, metres): the scripted scenario puts each fall in the same spot")
    print(f"  all chest readings: x standard deviation {chest['x'].std():.2f}, y standard deviation {chest['y'].std():.2f}")
    for n, g in places.groupby("fall"):
        print(
            f"  fall {n} of each recording ({len(g)} recordings): mean x {g['x'].mean():.2f} (sd {g['x'].std():.2f}),"
            f" mean y {g['y'].mean():.2f} (sd {g['y'].std():.2f})"
        )

    print("\n3. Chunk lengths: runs of one merged label, the units the HMM and HSMM classify")
    lengths = chunk_lengths(df, seqs)
    lam = {k: float(np.mean(v)) for k, v in lengths.items()}
    order = sorted(lengths, key=lambda k: np.median(lengths[k]))
    print(f"  {'class':<14}{'chunks':>7}{'median':>8}{'mean':>8}   frames")
    for k in order:
        print(f"  {k:<14}{len(lengths[k]):>7d}{np.median(lengths[k]):>8.0f}{lam[k]:>8.1f}")
    k = int(np.median(lengths["falling"]))
    print(f"\n  Poisson duration log-probability of a {k}-frame chunk (the median fall) under each class's mean length,")
    print("  the term the HSMM (cell 119) adds to the emission log-likelihood of the ground-truth chunk:")
    for c in order:
        print(f"  {c:<14}{log_poisson(k, lam[c]):9.1f} nats")
    others = max(log_poisson(k, lam[c]) for c in order if c != "falling")
    print(f"  'falling' leads the best other class by {log_poisson(k, lam['falling']) - others:.1f} nats on duration alone.")


if __name__ == "__main__":
    main()
