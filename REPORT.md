<h1 align="center">📱 Smart Home Fall Detection for Elderly Safety</h1>
<h3 align="center">Data-Driven Simulation</h3>
<hr>


## Overview 

Falls are a leading cause of injury among elderly individuals living independently.  

This project implements a **data-driven fall detection system** using motion sensor data collected from a smart home environment.  

Instead of relying on a single classification model, the system compares multiple sequence modeling approaches:

| Model | Type | Purpose |
|-------|------|---------|
| **LSTM (Long Short-Term Memory)** | Deep learning sequence model(Baseline) | Sequence-aware fall detection |
| **Hidden Markov Model (HMM)** | Generative probabilistic | General activity modeling |
| **Random Forest + HMM Smoother** | Ensemble | Frame-level classification + temporal smoothing |
| **Hidden Semi-Markov Model (HSMM)** | Probabilistic with duration prior | Safety-critical fall detection |

The frontend provides an **interactive Streamlit dashboard** that performs sliding-window fall detection, visualizing fall probability over time.

---

## About the Dataset

### Context
This dataset originates from research on safer smart environments for independent living. It was collected to analyze human activity using **body-worn sensors**, with a focus on understanding movements and detecting falls in a controlled setting.

### Dataset Overview
- **Participants:** 5 individuals  
- **Repetitions:** Each participant performed the same scenario **five times**  
- **Sensors:** Four body-worn tags per participant:
  - `ANKLE_LEFT` → 010-000-024-033  
  - `ANKLE_RIGHT` → 010-000-030-096  
  - `CHEST` → 020-000-033-111  
  - `BELT` → 020-000-032-221  
- **Measurements:** 3D localization coordinates (`X`, `Y`, `Z`) for each sensor  
- **Activities:** walking, falling, lying down, sitting down, standing up, on all fours, and others (11 total)  
- **Instances:** 164,860  
- **Features:** 8 (sequence name, tag ID, timestamp, date, X, Y, Z, activity label)  
- **Missing values:** None  

### Data Structure
- Each row corresponds to a **single sensor reading** at a specific timestamp.  
- Sequence name identifies the participant and repetition: e.g., `A01` → Participant A, first scenario repetition.  
- Tag ID identifies the sensor.  
- `Activity` column is categorical with 11 possible activities.

### Variable Summary

| Column        | Description |
|---------------|-------------|
| Sequence Name | Participant & repetition (A01–E05) |
| Tag ID        | Sensor identifier (ANKLE_LEFT, ANKLE_RIGHT, CHEST, BELT) |
| Timestamp     | Unique numeric timestamp |
| Date          | Format: `dd.MM.yyyy HH:mm:ss:SSS` |
| X             | X-coordinate of the tag |
| Y             | Y-coordinate of the tag |
| Z             | Z-coordinate of the tag |
| Activity      | Human activity label (walking, falling, lying, etc.) |


### Acknowledgements
Refactored from the [UCI Localization Data for Person Activity dataset](https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity).  
> B. Kaluza, V. Mirchevska, E. Dovgan, M. Lustrek, M. Gams, *An Agent-based Approach to Care in Independent Living*, International Joint Conference on Ambient Intelligence (AmI-10), Malaga, Spain, In press.

---

## Project Structure
```
project_root/
│
├── data/
│   └── ConfLongDemo_JSI.txt
│
├── model/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── inference.py
│   │
│   ├── hmm/
│   │   ├── __init__.py
│   │   └── train.py
│   │
│   ├── lstm/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── train.py
│   ├── rf/
│   │   ├── __init__.py
│   │   └── train.py
├── app.py
├── requirements.txt
└── README.md
```

---
## Exploratory Data Analysis 
### Acceleration Magnitude Distribution

Acceleration magnitude summarizes the overall motion intensity captured by the wearable sensors. The chart below shows the distribution helps identify differences in movement patterns across sensor locations.

<p align="center">
  <img src="asset/acceleration_magnitude.png" width="650">
</p>

Key observations:

- The ankle readings have higher acceleration magnitudes, corresponding to more dynamic movements.
- The chest and belt sensors show lower values, which are associated with stationary or low-motion activities.

---

### Activity Transition Patterns

To understand how activities evolve over time, we computed transition probabilities between activities using a normalized transition matrix.

<p align="center">
  <img src="asset/activity_transitions.png" width="800">
</p>

Key observations:

- Certain activities exhibit strong self-transitions, such as lying and sitting, indicating temporal persistence.
- These patterns motivate the use of **sequence models such as Hidden Markov Models (HMMs)**.

---

### Sensor Signal Over Time

The following chart shows raw accelerometer signals along the X, Y, and Z axes for a single sequence A01. 

<p align="center">
  <img src="asset/sensor_signal.png" width="700">
</p>

Key observations:
- Smooth signal segments indicate periods of stable activity, while sudden spikes may correspond to abrupt movements. The signal trend shifting show potential activites changes

---
### Accelation Time Series by Participants

The following chart shows the raw accelerometer signals recorded from participants **A–E** across their repeated activity sessions.

<p align="center">
  <img src="asset/acceleration_ts_by_participant.png" width="700">
</p>

Key observations:
- Each participant performs the same scenario **five times**, but the sensor patterns vary across repetitions.
- Because the signal patterns differ between repetitions, it may be more appropriate to **treat each repetition as an independent sequence** during modeling.


---
### Sensor Reading by Activities

The following chart shows raw accelerometer signals across different **activities**.

<p align="center">
  <img src="asset/sensor_readings_by_activity.png" width="700">
</p>

Key observations:
- Some activities produce **very similar sensor patterns**, such as *sitting down*, *sitting*, and *sitting on the ground*.
- These activities may potentially be **grouped into broader activity categories** during modeling or preprocessing.






---
## Data Preprocessing

**General Preprocessing Steps For All Models:**

- Load dataset and assign descriptive column names  
- Convert `date_time` to pandas datetime and acceleration columns to float  
- Sort by `sequence_name` and `date_time` for chronological order  
- Check for missing values and verify data integrity  
- Map activity labels for binary or multi-class tasks  
- Apply **StandardScaler** for feature normalization  
- One-Hot encode categorical features (e.g., sensor ID)  
- Group by `sequence_name` for participant-level analysis  
- Compute derived features like **acceleration magnitude**

**Distinctive Preprocessing:**
- **Random Forest / HMM:**
  - Activities combined into 6 macro-activities (walking, falling, lying, sitting, standing, on all fours)
  - Sliding windows → statistical features per window (mean, std, min, max, skew, kurtosis)  
- **HMM / HSMM:**
  - 6 macro-activities as in Random Forest + HMM  
  - Sequence-level multivariate observations, optionally augmented with duration features for HSMM

---

<img width="869" height="504" alt="image" src="https://github.com/user-attachments/assets/c102bc89-7eec-46d1-8e89-de826f1c9b49" />


---

## Backend Models

### 1. LSTM (Baseline)

- **Bidirectional 2-layer LSTM** with hidden size `128`, capturing temporal patterns in **12-dimensional motion features** for each frame sequence.

- Sequence outputs `h_t` are fed into a **linear classifier**:  `z_t = W * h_t + b`  producing **frame-level logits**.

- **Sigmoid activation** converts logits to fall probabilities:  `p_t = 1 / (1 + exp(-z_t))`  with an optimized **decision threshold (~0.94)** for detecting falls.

- Trained using **class-weighted Binary Cross-Entropy with logits (BCEWithLogitsLoss)** and the **Adam optimizer**, with the positive class weight calculated per training fold to handle severe class imbalance.

### 2. 12D HMM

- **One Gaussian HMM per activity class** (`falling`, `lying`, `sitting`, `standing`, `walking`, `on_all_fours`) trained on **12D motion features** from four body sensors (`ANKLE_LEFT`, `ANKLE_RIGHT`, `BELT`, `CHEST`).

- **Activity sequences are split into chunks** at label changes; features are **standardized** before training. Each HMM has **3 hidden states with full covariance matrices** to model temporal dynamics.

- **Chunk-level predictions** are computed using **length-normalized log-likelihoods**, converted to probabilities via **softmax**. The predicted activity is the class with the highest probability.

- Predictions are **expanded to frame-level**, providing a **falling probability per frame** for threshold-based detection and evaluation with frame-level accuracy and macro F1-score.

### 3. Random Forest + HMM Smoother
- Stage 1: RF predicts frame-level activity probabilities from sensor features  
- Stage 2: HMM smooths RF predictions using the **Viterbi algorithm**  
- Combines nonlinear frame-level classification with **temporal coherence**
- **Leave-One-Sequence-Out (LOSO)** training and evaluation where the full pipeline is rerun independently for each of the 25 folds (sequences) 

### 4. Hidden Semi-Markov Model (HSMM) – Not in Dashboard
- Two levels: **Gaussian HMM** as an emission sub-model combined with an explicit **Poisson duration distribution**
- Each frame is represented by an **8-dimensional feature vector** derived from 4-body worn sensors
    - 4 magnitude features
    - 4 jerk features
- Per-activity Gaussian HMM to model each activity's internal dynamics
- 6 x 6 activity-level transition matrix is estimated by counting chunck-to-chunck transitions with Laplace smoothing (no self-transitions)
- Uses **segment-level Viterbi decoding**
- **Leave-One-Sequence-Out (LOSO)** training and evaluation for 25 folds


---
## Model Results

### 1. Baseline LSTM (Weighted)   
| Class         | Precision | Recall | F1-score | Support  |
|---------------|:---------:|:-----:|:--------:|---------:|
| Non-fall (0)  | 1.00      | 0.99  | 0.99     | 161,887  |
| Fall (1)      | 0.63      | 0.73  | 0.68     | 2,973    |
| **Overall**   | -         | -     | 0.99     | 164,860  |

  - Achieved **high overall accuracy (0.987)** and **macro F1 score (0.84)**

  - **Class imbalance was explicitly addressed** using **class-weighted Binary Cross-Entropy loss**, giving higher penalty to missed fall events during training.

  - The model achieved **fall detection performance of F1 = 0.68** with **recall = 0.73**, indicating it is able to capture many true fall events.

  - However, the model is **highly biased toward the dominant non-fall class** because fall frames represent only a **very small fraction of the dataset**.

  - As a result, the **extremely high overall accuracy could be misleading**, and the model may **struggle to generalize to new subjects or real-world environments** where fall patterns and class distributions differ.

--- 
### 2. Activity-Level 12D HMM Model


| Class        | Precision | Recall | F1-score | Support |
|--------------|:---------:|:------:|:--------:|--------:|
| Falling      | 0.10      | 0.29   | 0.15     | 2,973  |
| Lying        | 0.63      | 0.51   | 0.56     | 60,648 |
| On all fours | 0.00      | 0.00   | 0.00     | 5,210  |
| Sitting      | 0.40      | 0.47   | 0.43     | 40,729 |
| Standing     | 0.16      | 0.30   | 0.20     | 22,590 |
| Walking      | 0.69      | 0.31   | 0.43     | 32,710 |

**Accuracy:** 0.413  | **Macro Avg F1:** 0.296  | **Weighted Avg F1:** 0.43  
 
- Better-performing in **walking**, **lying** (moderate).  
- Poorly detected **falling, standing, and transitional activities** (e.g., on all fours).  
- Performance is limited by **class imbalance** and **rare-event sequences**, affecting generalization to infrequent motions.



---


### 3. Random Forest (RF) + HMM Smoother


<p align="center">
  <img src="asset/RFHMM_ConfusionMatrix.png" width="650">
</p>



- Achieved the **highest overall accuracy (0.85)** and **macro F1 score (0.74)** across all tested models, outperforming the best previous pure HMM (**12D model: 0.72 accuracy**) by a meaningful margin.

- The **HMM smoothing layer improved accuracy by +0.02** over the Random Forest alone (**0.828 → 0.847**) by penalizing physically implausible frame-to-frame transitions.

- These results confirm that **temporal context adds value** on top of frame-level classification.

- **Walking** and **sitting** are classified with high reliability (**F1 = 0.93** and **0.92**, respectively), reflecting their distinctive and consistent sensor signatures across subjects.

- **Falling remains the weakest class** (**F1 = 0.48**, **recall = 0.33**), meaning the model misses roughly **two-thirds of actual fall events** at the frame level. This represents a critical limitation for a fall detection system, where **false negatives carry a high safety cost**.

- The **precision–recall imbalance for falling** (**precision = 0.86**, **recall = 0.33**) suggests the model is **conservative**: when it predicts a fall it is usually correct, but it fails to flag the majority of actual falls. This is likely because falls occupy **very few frames relative to the total dataset**.

<p align="center">
  <img src="asset/RFHMM_Normalized_ConfusionMatrix.png" width="700">
</p>

**Well-Classified Activities:**

- **Walking** is classified near-perfectly with a recall of 0.98, meaning the model correctly identifies 98% of all walking frames. Its sensor signature (rhythmic, periodic oscillation across all four body positions) is highly distinctive and consistent across subjects
- **Sitting and lying** achieve 0.89 and 0.88 recall respectively, also well-classified. The low acceleration magnitude and stability across all sensors makes them easy to separate from dynamic activities


**Problematic Activities:**

- **Falling has the lowest recall at 0.33**, meaning the model misses two-thirds of actual fall events. Misclassifications are spread across all other classes — walking (0.27), standing (0.18), lying (0.11), and sitting (0.12) — suggesting the model sees different phases of a fall and assigns each phase to whichever static or dynamic activity it most resembles at that frame. This may be a consequence of frame-level classification without duration awareness

- Falling is confused with many classes, but no other class is significantly confused with falling (all off-diagonal entries in the falling column are 0.00 or very small). This means the model **rarely produces false fall alarms** but **very frequently misses true falls** — a precision-recall tradeoff that is the opposite of what a safety-critical fall detection system should aim for

<p align="center">
  <img src="asset/RFHMM_Top20Features.png" width="700">
</p>

- **CHEST_z is the single most important feature by a large margin**, with a mean decrease in impurity of ~0.135 — more than double the next features. The vertical axis of the chest sensor captures postural orientation most directly, making it the strongest individual discriminator between upright activities (walking, standing) and horizontal ones (lying, falling)
  
- Chest and belt sensor dominate feature importance (CHEST_roll_mean, CHEST_x, BELT_z, BELT_x, CHEST_y, and BELT_y all appear in the top 7), confirming that **the torso-mounted sensors (chest and belt) collectively carry the most discriminative signal**. This is expected since trunk orientation changes most dramatically across activities

- ANKLE_LEFT_roll_mean and ANKLE_RIGHT_roll_mean both rank higher than their raw axis counterparts. This indicates that **temporal smoothing adds more information for ankle sensors** than for chest/belt sensors, likely because ankle readings are noisier and the rolling mean reduces variance to expose the underlying activity state

- Magnitude features, including mag_mean and CHEST_mag, ANKLE_RIGHT_mag, BELT_mag, ANKLE_LEFT_mag, all appear in the top 20. Magnitude is rotation-invariant, so it provides reliable signal regardless of how a tag is oriented on the body

  
---

### 4. Hidden Semi-Markov Model
<p align="center">
  <img src="asset/HSMM_ConfusionMatrix.png" width="700">
</p>



  - The model performs **poorly overall**, with a macro F1 of **0.19** and overall accuracy of **21%**, which is barely above chance for a 6-class problem. However, the results are not uniform across classes.
    
  - The Falling Class is the **best performing class** in the model with a precision of 0.68, meaning relatively few false alarms, and recall of 0.40 which misses 60% of true falls.
    
  - Like the RF + HMM model, low recall is still the problem here. **The model is conservative**, reluctant to predict "falling" which gives decent precision but at the cost of too many missed events.
    
  - For a safety applicatio, we would want to optimize **recall over precision** to catch more real falls, so this trade-off is currently the wrong way around.
    
  - The remaining five classes tell a concerning story. Walking is predicted everywhere, lying and sitting are completely missed. This pattern suggests the model has learned that **"walking" like acceleration magnitudes and jerks are a safe default**, while the low-movement activities are consistently outscored.
    
  - The poor performance of HSMM could be attributed to severe class imbalance and feature similarities among static activities. In a way, lying, sitting, and standing all produce low-magnitude, low-jerk signals. A richer feature set (e.g., frequency-domain features) would help separate these. 

---

## Summary 


| Model           | Overall Accuracy | Falling Recall | Falling F1 | Notes |
|-----------------|----------------|----------------|------------|-------|
| LSTM (Weighted) | 0.99           | 0.73           | 0.68       | Binary model; highest overall accuracy but biased on falling due to severe class imbalance |
| 12D HMM (fall)        | 0.10           | 0.29           | 0.15       | Baseline HMM using all 12 sensor features; lowest overall performance |
| RF + HMM (fall)        | 0.86           | 0.33           | 0.48       | Frame-level RF predictions smoothed by HMM; high multi-class accuracy |
| HSMM (fall)            | 0.21           | 0.40           | 0.50       | Temporal model focused on fall events; moderate fall detection, lower overall accuracy

- **LSTM (Weighted)** achieves **very high overall accuracy (0.99)**, but this is largely driven by the **dominant non-fall class**. While it captures many fall events (**recall 0.73, F1 0.68**), the model is **less reliable for unseen subjects or real-world falls** but can serve as premilary fall trigger

- **12D HMM** uses all 12 sensor features as a baseline and achieves **low frame-level performance (accuracy 0.41, macro F1 0.30)**. It detects common activities like sitting, walking, and lying moderately well but struggles with rare or transitional events such as falling and being on all fours.

- **RF + HMM (fall)** combines frame-level Random Forest predictions with HMM smoothing. This approach provides **the best multi-class accuracy (0.85)** and **temporal consistency**, though its fall detection performance remains modest (**recall 0.33, F1 0.48**).

- **HSMM (fall)** results in **low overall accuracy (0.21)** and **moderate fall detection (recall 0.40, F1 0.50)**. While it incorporates **explicit temporal modeling to capture fall events**, its performance is **limited compared to other models**, reflecting the trade-off between fall sensitivity and overall accuracy.

### Potential Implementation


**Option 1:** Combine predictions from multiple models to create a **more robust fall alert**, potentially with **custom logic**, though this may increase computational complexity.


**Option 2:** Use **LSTM (Weighted)** as a **initial fall alert**, while leveraging **RF + HMM Smoother** for general activity state **modification**.


---

## Limitation and Future Improvement

- **Enhanced sensors:** Add gyroscopes, pressure sensors, or ambient devices for richer motion data.  
- **Class imbalance & bias:** LSTM favors non-fall frames; use data augmentation or class-weighted losses to improve fall recall.  
- **Generalization & validation:** Test on larger, diverse datasets for robust real-world performance.  
- **Interpretability:** Add explainability (e.g., SHAP, attention) to increase trust in predictions.  
- **Multi-model approach:** Combine HSMM (high fall recall) and RF+HMM (multi-class accuracy) for a robust fall alert system.

---
Sample Use
---
### 📱 Multi-Model Fall Alert

For each sliding window of sensor data, three models produce fall-related prodictions. The final fall probability is computed using a **weighted ensemble**. The weights were chosen empirically based on model performance, giving higher influence to the LSTM due to its stronger fall detection capability while still incorporating complementary signals from the RF+HMM and 12D HMM models:

```
p_ensemble = 0.55 * p_LSTM
           + 0.30 * p_RF
           + 0.15 * p_HMM
```

A fall alert is triggered using a threshold decision rule:

```
if p_ensemble > 0.5:
    prediction = "falling"
else:
    prediction = "non-fall"
```

```mermaid
flowchart LR

A["Wearable Sensors\nAnkle L/R, Belt, Chest"] --> B["Sensor Data Stream"]
B --> C["Preprocessing Pipeline"]
C --> D["Sliding Window Segmentation\nWindow=100, Step=50"]

D --> E["LSTM Model\nFall Probability"]
D --> F["RF + HMM Smoother"]
D --> G["12D HMM Model\nBaseline"]

E --> H["Ensemble Fall Decision"]
F --> H
G --> H

H --> I["Final Fall Alert"]
I --> J["Streamlit Dashboard\nVisualization"]

```
### Dashboard Demo
<p align="center">
  <img src="asset/demo.gif" width="700" alt="Fall Detection Dashboard Demo"/>
</p>

---



## Running the Application

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```
**2. Train models (only if not already trained):**

- Train hmm model
```bash
python -m models.hmm.train
```

- Train lstm model - This might take a long time
```bash
python -m models.lstm.train
```

- Train RF+HMM model
```bash
python -m models.rf.train
```

**3. Run the Streamlit app**

```bash
streamlit run app.py
```

---
### Interact with the Fall Detection Dashboard

The dashboard provides an interactive interface for exploring participant sensor sequences and visualizing fall detection results.

**Features:**

- **Select participant sequences:**  
  Choose from sequences containing falls to analyze specific events.

- **Run sliding window detection:**  
  Execute LSTM, Random Forest, and HMM models on fixed-size sliding windows of sensor data.

**Visualizations:**

- **Sensor readings over time:**  
  View all accelerometer and gyroscope signals. Ground truth fall periods are marked with **red vertical lines**:  
  - Solid line → Fall start  
  - Dashed line → Fall end

- **LSTM & Random Forest fall probability:**  
  Window-level probabilities plotted as line charts. High-risk windows (predicted as falling) are highlighted in tables.

- **HMM activity predictions:**  
  Model-predicted activity classes over time are displayed as line plots.

- **Essembled fall alerts:**  
  Identify windows where **any model predicts a fall**, summarized in a table for quick review.

The interface allows users to explore **ground truth vs. model predictions** interactively, making it easier to validate and analyze fall detection performance.


