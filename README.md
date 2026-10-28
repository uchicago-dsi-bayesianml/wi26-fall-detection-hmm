<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Smart Home Fall Detection for Elderly Safety

> **Example project from Winter 2026.** This is a copy of [XjFa/fall-detection-system](https://github.com/XjFa/fall-detection-system), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project). The recordings are UCI's "Localization Data for Person Activity", public under CC BY 4.0 and kept in this repository with attribution; [data/README.md](data/README.md) describes them.

An older person who lives alone and falls may not be able to call for help, and a fall alarm is only useful if it catches most falls without going off so often that nobody answers it. Can a model that knows how activities follow one another tell a fall from lying down, sitting or kneeling, using four tags worn on the body? The team took 25 recordings of five people acting out the same scripted scenario, with tags on both ankles, the belt and the chest, and built four models: one Gaussian hidden Markov model per activity, a random forest whose frame-by-frame class probabilities are smoothed by Viterbi decoding with an activity transition matrix, a hidden semi-Markov model with Poisson durations, and a bidirectional LSTM as a baseline. A Streamlit dashboard runs three of them over sliding windows of a recording and raises an ensemble alarm.

**A note on terms.** The tags are located by an ultra-wideband radio system, and each row of the data is one tag's position in the room, in metres. Nothing in this data measures acceleration. The write-up and several of its figures call the coordinates accelerometer signals, the length of a tag's position vector "acceleration magnitude" and its change between frames "jerk". Read these as position, distance from the room's origin, and change in that distance.

## Summary

**Data.** UCI dataset 196 from the Jožef Stefan Institute: 164,860 rows, each one tag's x, y and z position with a time stamp and one of eleven activity labels. Five people each performed the same scenario five times, giving 25 recordings and 91.0 minutes. Every recording contains three falls, 75 in all, lasting 1.26 seconds on average; 1.8 percent of rows are labelled falling. The team merges the activities into six: walking, falling, lying, sitting, standing (the three standing-up transitions) and on all fours.

**Frames and features.** The tags are read in turn, about 27 ms apart, and each row becomes a frame of twelve coordinates by carrying the other tags' last positions forward. The forest adds per-tag vector lengths, their changes and ten-frame rolling statistics, and four cross-tag summaries: 32 features.

**Models.** Per-activity Gaussian HMMs, fit in hmmlearn, label a stretch of frames with the activity whose HMM fits it best, on the four vector lengths ("4-D") or the twelve coordinates ("12-D"). The forest gives class probabilities for every frame, and Viterbi with a transition matrix counted from the training labels turns them into the most probable label path. The HSMM adds a Poisson log-probability of a stretch's length. The LSTM labels each frame fall or not fall.

**Evaluation.** Leave-one-sequence-out cross-validation over the 25 recordings ("LOSO"), scored with frame-level classification reports.

**Results as reported.** The smoothed forest has the best six-class accuracy, 0.847 against 0.828 without smoothing, with fall recall 0.33; the weighted LSTM has the best fall F1, 0.68 at recall 0.73. The write-up suggests the LSTM as the first alarm and the smoothed forest for tracking activity.

**The dashboard.** `app.py` slides a 100-frame window over a recording and raises an alarm when a weighted average of three models' outputs exceeds 0.5.

The staff section below shows that the smoother is the project's strongest result, undersold because the write-up scores frames rather than alarms, and which other numbers rest on shortcuts: test segments cut at the true activity boundaries, a duration model handed the true durations, a threshold tuned on the test folds, and falls that always happen in the same three places.

## Team and contributions

Team "Latent Space Warriors", Winter 2026.

- **Xiaojing Fang** ([@XjFa](https://github.com/XjFa)): the repository and nearly all of its code: every version of the notebook, now `notebooks/model_fitting.ipynb`, including the random forest and HSMM sections that first appear in the version of February 26; the training modules under `models/` and `models/inference.py`; the Streamlit app and its demo GIF; the exploratory figures; and the write-up's structure and its dataset, exploratory, dashboard and ensemble sections.
- **Lexi Lin** ([@lexi-lin](https://github.com/lexi-lin)): the write-up's sections on preprocessing, model details and results, including the analysis of the random forest and HSMM results, and the four result figures `RFHMM_ConfusionMatrix.png`, `RFHMM_Normalized_ConfusionMatrix.png`, `RFHMM_Top20Features.png` and `HSMM_ConfusionMatrix.png`.

The repository ran from February 16 to March 9, 2026, with 37 commits, 28 by Xiaojing Fang and 9 by Lexi Lin. The notebook was always committed from Xiaojing Fang's account, so the account that committed a cell is not necessarily its author; the random forest and HSMM cells differ in style from the rest, and their figures were committed by Lexi Lin. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/wi26-fall-detection-hmm/commits/main).

## Getting started

The data is in the repository and the notebook keeps its outputs, so it can be read without running anything. `requirements.txt` lists its packages without versions.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
```

**The pipeline.**

1. `notebooks/model_fitting.ipynb` is the project: data checks and exploratory figures, the HMMs, the random forest with the smoother, the HSMM and two LSTMs. It reads `../data/ConfLongDemo_JSI.txt`, so start Jupyter in `notebooks/`.
2. From the repository root, `python -m models.hmm.train`, `python -m models.lstm.train` (the authors warn that it may take a long time) and `python -m models.rf.train` train the app's models into `models/*/artifacts/`, which is git-ignored.
3. `streamlit run app.py` starts the dashboard.

**Staff check.** `staff/check_event_metrics.py` rebuilds the forest and the smoother from cells 108 to 111 and scores them as an alarm under the authors' split and a split by person, with and without features that depend on room position, and prints where the falls happen and the chunk lengths behind the HSMM's duration term. It needs NumPy, pandas and scikit-learn; with 50 trees per forest (the notebook uses 200) it ran in about 75 seconds on an 18-core laptop.

```sh
python staff/check_event_metrics.py        # --trees 200 matches the notebook, at four times the cost
```

The script defines its terms at the top, and this README uses them the same way. A *frame* is one row of the data file. A *fall event* is a maximal run of frames labelled falling in one recording; there are 75. A *fall segment* is a maximal run of frames a model labels falling. A fall is *caught* if some fall segment overlaps the fall widened by 40 frames, about one second, on each side. A *false alarm* is a fall segment that overlaps no widened fall. *Per hour* divides by the 91.0 minutes of recording.

**What does not run as committed.** Cell numbers count every cell of the notebook from the top, starting at 1. Cell 103 has a syntax error (`rep_4d =ort classification_report(...)`), so its stored output comes from an earlier version of the cell. Cell 100, behind the write-up's frame-level 12-D table, keys its state counts by the six merged labels but reads the eleven raw ones, so run in order it stops with a `KeyError`; its stored output, whose `nan` class of 5,210 frames the write-up relabels "On all fours", comes from a kernel state the committed cells do not recreate. The hierarchical HMM (cells 52 to 75) runs but does not do what it describes: hmmlearn overwrites the global parameters copied into each per-recording HMM (see its warnings in cell 57), and test recordings are labelled with another model's state mapping, for a fall F1 of 0.00 (cell 59). The input-output HMM in cell 128 stops with an `ImportError` inside seqlearn. The HSMM cells 118 to 120 have no stored output, cell 121 ("Full Viterbi Decoding") holds only imports, and cell 124 uses names no cell defines, such as `duration_params`; the write-up's HSMM figure is a screenshot of a report from a run not in the notebook. Run from the top, the LSTM cells would train on the forest's 32 features, because cell 109 reassigns `X_sequences`; their stored output (cell 134) shows twelve.

The app does not run the models the notebook evaluates. `models/rf/train.py` trains its forest on the 12 raw coordinates, because the sequences `preprocess_pipeline` returns hold only those, and saves the forest of its last fold, which never saw recording E05. `models/inference.py` averages the forest's fall probabilities over each window with no Viterbi smoothing, although the demo GIF, recorded from an earlier version of the app, titles that panel "Random Forest + HMM Smoothing". The HMM and LSTM are trained on all 25 recordings, so the dashboard only ever shows recordings they were trained on.

## Data

`data/ConfLongDemo_JSI.txt` is byte-identical to UCI's download; [data/README.md](data/README.md) gives the columns, tag IDs, time format, label counts, citation and download link.

- **Source:** [UCI Machine Learning Repository, dataset 196](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity), from the Jožef Stefan Institute's Confidence project, whose papers describe a Ubisense ultra-wideband location system sampling each tag at 10 Hz with a specified accuracy of 15 cm.
- **Terms:** CC BY 4.0. Cite the dataset and Kaluža et al., "An Agent-based Approach to Care in Independent Living", AmI 2010, LNCS 6439.
- **Signals:** x, y and z positions in metres, z being height; while walking, the median chest tag is 1.26 m above the floor (staff script).
- **Recordings and labels:** five people, five repetitions of one scripted scenario, eleven activities merged into six. There is no plain "standing" activity; the merged class is the three standing-up transitions.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up. It embeds the 11.8 MB `asset/demo.gif` and one image hosted outside the repository, a preprocessing diagram behind a GitHub user-attachments link. |
| `notebooks/model_fitting.ipynb` | Everything evaluated: exploratory figures, the hierarchical and per-activity HMMs, the forest with the smoother, the HSMM, the input-output HMM and two LSTMs. |
| `models/preprocessing.py` | Loading, tag mapping, the activity merge, the 12-coordinate frames and the 32 forest features. |
| `models/hmm/train.py`, `models/lstm/`, `models/rf/train.py` | Training of the app's three models, described under Getting started. |
| `models/inference.py`, `app.py` | Window-level predictions, the weighted ensemble and the Streamlit dashboard. |
| `asset/` | The write-up's figures and the demo GIF. |
| `data/` | The UCI data file, UCI's description and the staff data note. |
| `staff/check_event_metrics.py` | Course staff's check of the smoother as an alarm, described above. |

## Method

**Preprocessing.** Rows become twelve-coordinate frames as described above, and features are standardized with training-fold statistics. The hierarchical HMM and the LSTMs separate falling from everything else; the other models use the six merged labels.

**Per-activity HMMs.** Each training recording is cut into chunks wherever the label changes, and each activity's chunks train one Gaussian HMM with two to four hidden states (diagonal covariances for 4-D, full for 12-D). A test recording is cut the same way, at its own true label changes, and each chunk gets the activity whose HMM gives it the highest log-likelihood per frame. The frame-level 12-D run expands each chunk's label to its frames.

**Random forest with an HMM smoother.** A forest of 200 trees with balanced class weights is trained on the 32 features of every training frame and gives six class probabilities per test frame. A transition matrix is counted from consecutive training labels with one added to every cell, and a hand-written Viterbi pass (cell 110) with a uniform start finds the most probable label path, using the forest's probabilities as emission scores.

**Hidden semi-Markov model.** Each activity gets a Gaussian HMM on the four vector lengths and their changes, and a Poisson duration whose rate is the activity's mean chunk length in the training folds. A test chunk, again cut at the true label changes, is scored for each activity as the HMM log-likelihood plus the Poisson log-probability of its length. A 6 × 6 transition matrix between activities is counted but never used in scoring.

**LSTMs.** A two-layer LSTM with 64 hidden units, plain binary cross-entropy and a 0.5 threshold, and a two-layer bidirectional LSTM with 128 units whose loss weights fall frames by the ratio of other frames to fall frames. The bidirectional model's threshold, 0.94, maximizes F1 over the pooled predictions of all 25 test folds (cell 150).

**The dashboard.** For each window of 100 frames, in steps of 50, the alarm fires when 0.55 times the LSTM's mean fall probability plus 0.30 times the forest's plus 0.15 times the HMMs' falling probability exceeds 0.5. The write-up says the weights "were chosen empirically based on model performance"; the ensemble is not evaluated anywhere.

### What is Bayesian here

The per-activity HMMs are generative models of each activity's positions over time, fit with the forward-backward algorithm, and labelling a chunk by the highest likelihood is Bayes' rule with a uniform prior over activities. The "falling probability", a softmax of per-frame log-likelihoods, is a flattened version of that posterior.

The smoother is the most useful piece. It treats the six activities as the hidden states of an HMM, the forest's class probabilities as the evidence at each frame and the counted transition matrix as the prior over activity sequences, and Viterbi finds the most probable path, the decoding problem of the course's HMM sessions. A textbook hybrid divides a classifier's posterior by the class prior to get a likelihood; with balanced class weights the forest's outputs are already close to prior-free. Adding one to every count before normalizing each row of the transition matrix gives exactly the posterior mean of that row under a Dirichlet(1, …, 1) prior; the team calls it Laplace smoothing, which is the same thing.

The HSMM's Poisson durations are the explicit-duration idea, a prior over how long each activity lasts where an ordinary HMM implies a geometric one; as built, it scores chunks whose lengths come from the labels, so it decodes nothing.

What was not built: the methodology document promised a Bayesian HMM with priors on the transition and emission parameters, fit by variational Bayes or MCMC in Pyro, to quantify uncertainty. Everything in the repository is maximum likelihood in hmmlearn, with point estimates and no posterior over any parameter, and the dashboard's probabilities are forest vote shares and softmaxed scores, not calibrated posteriors. An alarm threshold is a decision with very different costs for a missed fall and a false alarm, which is what Bayesian decision theory is for; the thresholds here are 0.5 or the F1 maximum.

## Results

**As reported, checked against the notebook.** Every number matches the stored output of the cell given, except the HSMM row, which exists only in the write-up's figure. Accuracy and macro F1 are over six classes for the HMMs and the forest and over two for the LSTMs, so the groups cannot be compared on those columns.

| Model | Unit scored | Accuracy | Macro F1 | Fall precision | Fall recall | Fall F1 | Cell |
|---|---|---:|---:|---:|---:|---:|---|
| 4-D HMM | 700 true chunks | 0.591 | 0.528 | 0.44 | 0.61 | 0.51 | 103 |
| 12-D HMM, full covariance | 700 true chunks | 0.761 | 0.699 | 0.67 | 0.75 | 0.71 | 99 |
| 12-D HMM, another run | 700 true chunks | 0.716 | 0.654 | 0.59 | 0.68 | 0.63 | 103 |
| 12-D HMM, frame-level run | frames | 0.413 | 0.296 | 0.10 | 0.29 | 0.15 | 101 |
| Random forest | frames | 0.828 | 0.70 | 0.59 | 0.32 | 0.42 | 111 |
| Random forest + HMM smoother | frames | 0.847 | 0.74 | 0.86 | 0.33 | 0.48 | 111 |
| HSMM | frames | 0.21 | 0.19 | 0.68 | 0.40 | 0.50 | figure only |
| LSTM, one direction, threshold 0.5 | frames, binary | 0.980 | 0.577 | 0.35 | 0.11 | 0.16 | 139 |
| Bidirectional LSTM, weighted, threshold 0.94 | frames, binary | 0.987 | 0.836 | 0.63 | 0.73 | 0.68 | 151 |

The bidirectional LSTM's area under the precision-recall curve for falling is 0.70 (cell 152). The hierarchical HMM finds 17 of 538 fall frames in its five held-out recordings (cell 59).

**Corrections to the write-up.**

- The summary table gives the 12-D HMM an accuracy of 0.10, which is its fall precision; the accuracy is 0.413 (cell 101). It gives the smoothed forest 0.86; cell 111 says 0.847.
- The forest section compares itself with "the best previous pure HMM (12D model: 0.72 accuracy)". That 0.72 is chunk-level with chunks cut at the true labels (cell 103); at frame level the 12-D HMM has 0.413.
- The 12-D table's "On all fours" row is a `nan` class in cell 101, which that run never predicts.
- The limitations credit the HSMM with "high fall recall". Its 0.40 is below the LSTM's 0.73 and comes from the duration leak described below.
- The preprocessing section lists per-window skew and kurtosis and one-hot encoded sensor IDs; the code computes neither. The data is described as accelerometer and gyroscope signals; it is tag positions.

**Staff check: the smoother as an alarm.** From `staff/check_event_metrics.py` with 50 trees per forest, which moves the frame-level numbers by under 0.01 (0.841 against the notebook's 0.847 with the smoother, 0.820 against 0.828 without). "By person" holds out all five recordings of one person at a time. The 23 features without room position are each tag's height, its change and its rolling standard deviation, each tag's horizontal speed and its rolling mean, the chest and belt heights above the ankles, and the horizontal chest-to-ankle distance; none changes if a recording is shifted across the room.

| Split | Features | Model | Accuracy | Fall precision | Fall recall | Falls caught | False alarms per hour |
|---|---|---|---:|---:|---:|---:|---:|
| By recording (the authors') | the authors' 32 | forest | 0.820 | 0.52 | 0.38 | 60 of 75 | 155.5 |
| By recording | the authors' 32 | forest + smoother | 0.841 | 0.73 | 0.40 | 44 of 75 | 4.0 |
| By person | the authors' 32 | forest | 0.790 | 0.45 | 0.35 | 52 of 75 | 179.9 |
| By person | the authors' 32 | forest + smoother | 0.818 | 0.62 | 0.38 | 44 of 75 | 5.3 |
| By recording | 23 without room position | forest | 0.799 | 0.32 | 0.07 | 55 of 75 | 155.5 |
| By recording | 23 without room position | forest + smoother | 0.845 | 0.77 | 0.06 | 18 of 75 | 3.3 |
| By person | 23 without room position | forest | 0.791 | 0.27 | 0.06 | 48 of 75 | 168.7 |
| By person | 23 without room position | forest + smoother | 0.837 | 0.73 | 0.05 | 12 of 75 | 4.6 |

Without the smoother, many "caught" falls are stray predictions that happen to land within a second of a fall: the forest without room position has fall recall 0.07 and still "catches" 55.

**Staff check: chunk lengths.** The units the HMMs and the HSMM classify, in frames, from the same script.

| Activity | Chunks | Median length | Mean length |
|---|---:|---:|---:|
| falling | 75 | 34 | 39.6 |
| standing | 175 | 117 | 129.1 |
| walking | 200 | 136 | 163.6 |
| on all fours | 25 | 216 | 208.4 |
| lying | 151 | 471 | 401.6 |
| sitting | 74 | 530 | 550.4 |

For a 34-frame chunk, the median fall, the Poisson duration term is −3.1 nats under falling's rate and −52.4 nats under the next best, standing: a lead of 49.3 nats for falling before the HSMM has looked at a single position.

## What to learn from this project

Course staff added this section. The project compares four model families on a hard, realistic problem; the points below are about what its results can and cannot support.

- **Score the system the way it will be used.** A fall alarm is judged by the falls it catches and how often it goes off for nothing, not by the share of frames it labels correctly. On frames, the write-up finds that the smoother adds 0.02 to accuracy and 0.01 to fall recall. Counted as alarms, it separates an unusable system from a usable one: on the authors' split the forest alone raises 155.5 false alarms per hour and the smoothed forest 4.0, while catching 44 of the 75 falls instead of 60; split by person, 179.9 against 5.3. The transition matrix knows that one fall frame in the middle of a walk is improbable, which is what an alarm needs. This is the project's best result, and the write-up never measures it. Report falls caught and false alarms per hour, and choose the operating point from them.
- **Check whether the test data is segmented by its labels.** The per-activity HMMs and the HSMM never find where an activity starts or ends: each test recording arrives cut at its true label changes, and the models only name the pieces. The chunk accuracies of 0.72 to 0.76 answer "given that an activity runs from here to here, which is it?", a far easier question than finding a fall in a stream. Detection needs segmentation, for example Viterbi over one HMM whose states are the activities.
- **Beware a duration model that is handed the true duration.** The HSMM adds the Poisson log-probability of each chunk's length, and the length comes from the labels. Falls are by far the shortest chunks, a median of 34 frames against 117 or more for everything else, so a fall-length chunk gets 49.3 nats in favour of falling from its length alone. That is why falling is the HSMM's best class (precision 0.68) while its accuracy is 0.21. An explicit-duration HMM searches over segmentations, and there the duration prior is a genuine prior; the "segment-level Viterbi decoding" the write-up describes is not in the code.
- **Tune thresholds on training data only.** The LSTM's threshold of 0.94 maximizes F1 over the pooled test-fold predictions (cell 150), so its F1 of 0.68 and recall of 0.73 are optimistic; the threshold-free PR area of 0.70 (cell 152) is the fair summary. A bidirectional network also reads future frames, so an alarm needs a causal model or a stated delay.
- **Hold out people, not recordings.** "LOSO" here means leave one recording out, so the same person's other four recordings are always in training, although cell 78 calls it leave-one-subject-out. The effect is modest here: holding out people lowers the smoothed forest's accuracy from 0.841 to 0.818 and leaves 44 falls caught. A claim about new people should still rest on the person split.
- **Ask whether the model knows what happened or where.** Every recording follows the same script in the same room, and each of its falls happens in the same place: across the 25 recordings, the chest's mean position during a given fall has a standard deviation of at most 0.26 m in x and 0.13 m in y, against 1.19 m and 0.51 m for the chest overall. In the write-up's feature-importance figure, four of the seven most important features are x or y coordinates of the chest and belt. With every feature that depends on room position replaced, the smoothed forest's accuracy stays at 0.845, but its fall recall drops from 0.40 to 0.06 and it catches 18 falls instead of 44, 12 when split by person. Much of the fall detection is the model knowing where the script puts the falls; better location-free features might recover some of it.
- **Name the signals correctly.** Calling positions accelerations led the write-up to read the distance from the room's origin as motion intensity and to suggest gyroscopes "for richer motion data" for data with no motion sensor at all. The team's methodology document, and a stick figure drawn from "absolute joint positions" in the first commit, had it right.
- **Make the demo run the model you evaluated.** The dashboard's forest uses 12 coordinates instead of 32 features and has no smoother. Its LSTM compares a window's mean probability with a threshold tuned for single frames, but a fall averages 40 frames, so a 100-frame window's mean rarely reaches 0.94. The demo GIF shows one flagged window in a recording with three marked falls, above a fixed line reading "True label: falling".
- **Ship the plan or revise it in writing.** The methodology document promised a Bayesian HMM with posterior inference in Pyro; the repository has maximum-likelihood HMMs, a broken hierarchical HMM and an input-output HMM that never ran. Dropping a plan is normal, but the write-up should say so.

What the project does well: an openly licensed dataset with a clear safety motive; four model families compared on the same frames with scaling fit inside each fold; class imbalance treated as the central problem, with class weights, a weighted loss, macro F1 and a precision-recall curve; a hand-written Viterbi smoother over a classifier's probabilities, the right Bayesian idea for this problem, which works when measured as an alarm; modular training code and an interactive dashboard; and a write-up that says plainly that high accuracy is misleading and that the models miss too many falls.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code is unchanged.

- Extended the authors' `.gitignore`, which already excluded `*.pkl` and `*.pt`, with the `models/*/artifacts/` folders the training scripts write, the PNG files the notebook's plotting cells save into `notebooks/`, and OS and virtual-environment clutter. The tree held no clutter files, so nothing was removed.
- Added `data/README.md`, on the dataset, its terms and citation and what its columns really are, and `staff/check_event_metrics.py`, the check described above.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner. `REPORT.md` keeps its wrong terms, its 11.8 MB demo GIF and its externally hosted preprocessing diagram as the authors committed them.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it. The recordings are CC BY 4.0, as above.
