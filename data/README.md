# Data

Both files in this folder are the UCI Machine Learning Repository's dataset #196, "Localization Data for Person Activity", as published. `ConfLongDemo_JSI.txt` is byte-identical to the copy in UCI's download (SHA-256 `1fbc965b8d7faf499ea854428b2138132d94e6a26404bc8e4b0c62144f513e8a`); `dataSetDescription.names` is UCI's description with Windows line endings changed to Unix ones and is otherwise identical.

## What the signals are

The four signals per person are **positions, not accelerations**. Each subject wore four radio tags, and a location system fixed with anchors in the room reported where each tag was. The dataset comes from the Jožef Stefan Institute's Confidence project (the file name is short for "Confidence long demo"), whose papers describe a Ubisense ultra-wideband real-time location system that samples each tag at 10 Hz with a specified accuracy of 15 cm, and larger deviations in practice (Luštrek et al., "Detecting Falls with Location Sensors and Accelerometers", IAAI 2011). There is no accelerometer or gyroscope in this data. The authors' write-up, [REPORT.md](../REPORT.md), and several of its figures call the coordinates "acceleration" and the norm of a tag's position vector "acceleration magnitude"; that norm is the tag's distance from the room's origin.

UCI does not state units. The values are consistent with metres: while walking, the median height of the ankle tags is 0.14 and 0.15, the belt tag 0.85 and the chest tag 1.26 (printed by `staff/check_event_metrics.py`).

## The file

`ConfLongDemo_JSI.txt` is comma-separated with no header: 164,860 rows, one per tag reading, 21.5 MB.

| Column | Contents |
|---|---|
| 1. Sequence name | `A01` to `E05`: person A to E, repetition 1 to 5 |
| 2. Tag ID | `010-000-024-033` left ankle, `010-000-030-096` right ankle, `020-000-032-221` belt, `020-000-033-111` chest |
| 3. Timestamp | A unique integer per row. It decodes as .NET ticks (units of 100 ns since 1 January 0001) two hours behind the date column, which looks like UTC against local summer time. The authors use the date column. |
| 4. Date | `dd.MM.yyyy HH:mm:ss:SSS`, for example `27.05.2009 14:03:25:127` |
| 5 to 7. x, y, z | The tag's position; z is height above the floor |
| 8. Activity | One of eleven labels, below |

The four tags are read in turn, so consecutive rows usually belong to different tags. Rows are 27 ms apart at the median (notebook cell 18), and each tag appears about every tenth of a second. The authors' pipeline turns each row into a 12-dimensional frame by carrying every other tag's last reading forward within the sequence (`models/preprocessing.py`).

**People and recordings.** Five people each performed the same scripted scenario five times, which gives 25 recordings and 91.0 minutes in all, about 3.6 minutes each. Every recording contains exactly three falls, 75 in total, lasting 1.26 seconds on average. The three falls of a recording happen at the same places in the room in every recording (see the staff README).

**Labels.** UCI's eleven activities and the authors' merge into six, with row counts from `dataSetDescription.names`:

| Recorded activity | Rows | Merged label | Rows |
|---|---:|---|---:|
| walking | 32,710 | walking | 32,710 |
| falling | 2,973 | falling | 2,973 |
| lying, lying down | 54,480 + 6,168 | lying | 60,648 |
| sitting, sitting down, sitting on the ground | 27,244 + 1,706 + 11,779 | sitting | 40,729 |
| standing up from lying, from sitting, from sitting on the ground | 18,361 + 1,381 + 2,848 | standing | 22,590 |
| on all fours | 5,210 | on_all_fours | 5,210 |

Falling is 1.8 percent of rows. There is no "standing" activity in the recording; the merged "standing" class consists of the three standing-up transitions.

## Source, terms and citation

- **Source:** [UCI Machine Learning Repository, dataset 196](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity), DOI [10.24432/C57G8X](https://doi.org/10.24432/C57G8X). Creators: Vedrana Vidulin, Mitja Luštrek, Boštjan Kaluža, Rok Piltaver and Jana Krivec, Jožef Stefan Institute, Ljubljana. Donated in 2010 by Božidara Cvetković.
- **Terms:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which allows the copy kept here with attribution. Cite the dataset and the paper below if you use it.
- **Paper:** B. Kaluža, V. Mirchevska, E. Dovgan, M. Luštrek and M. Gams, "An Agent-based Approach to Care in Independent Living", International Joint Conference on Ambient Intelligence (AmI 2010), Málaga, Spain, Lecture Notes in Computer Science 6439, Springer, 2010. The description file quotes its fall-detection results: 72 percent for the machine-learning agents, 88 percent for the expert-knowledge agents and 91.3 percent for the meta-prediction agents.
- **Dataset citation:** Vidulin, V., Luštrek, M., Kaluža, B., Piltaver, R., and Krivec, J. (2010). Localization Data for Person Activity [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57G8X.

The team's topic proposal named a Kaggle mirror of the same recordings, "Anomaly Detection: Falling People Events", which splits them into 25 CSV files. The repository uses UCI's original file.

## How to get it again

```sh
curl -L -o uci196.zip "https://archive.ics.uci.edu/static/public/196/localization+data+for+person+activity.zip"
unzip uci196.zip -d data/
```

The archive holds the same two files.
