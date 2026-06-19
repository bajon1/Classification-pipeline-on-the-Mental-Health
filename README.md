# Mental Health in Tech — MLP Classification Pipeline

This project implements a multiclass classification pipeline based on the OSMI Mental Health in Tech Survey (2014) dataset. The goal is to predict **work_interfere** — how strongly a mental health condition affects an employee’s work performance. Using a PyTorch MLP model with Optuna hyperparameter optimisation. The project covers the full machine learning workflow, including data preprocessing, feature engineering, model tuning, and evaluation on a real-world mental health dataset.

---

## Why `work_interfere` and not `treatment`?

The most intuitive prediction target in this dataset is `treatment`, whether a respondent has sought mental health treatment. However, `treatment` is a poor modelling target for two reasons.

The first is **data leakage**. Several features in the survey (`mental_health_consequence`, `mental_health_interview`, `benefits`, etc.) describe a respondent's *perception* of mental health support at their workplace. These perceptions are directly shaped by whether a person has already sought treatment, someone who has gone through the process of finding care will have far more informed and often more negative opinions about workplace attitudes. Using these as features to predict `treatment` would mean the model learns a near-tautological relationship rather than genuine signal.

The second is **task triviality**. The leakage makes `treatment` binary classification artificially easy, with published notebooks on this dataset regularly exceeding 80% accuracy. This is not a useful result as it reflects the circular structure of the survey rather than any real predictive insight.

`work_interfere` avoids both problems. It is a four-class ordinal target (Never / Rarely / Sometimes / Often) that captures the *functional impact* of mental health on work performance, which is a genuinely harder and more meaningful prediction task. It is not directly asked about by other features, so no feature leaks its value. It also produces a naturally imbalanced distribution, making it a more realistic and instructive classification problem.

---

## Project Structure

```
├── data/
│   ├── raw.csv
│   ├── cleaned.csv          # post-cleaning, pre-encoding (used for EDA)
│   ├── processed.csv        # one-hot encoded (used for modelling)
│   └── tensors/             # train/test splits as .pt tensors
│       ├── X_trainval.pt
│       ├── X_test.pt
│       ├── y_trainval.pt
│       ├── y_test.pt
│       └── sample_weights.pt
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_architecture.ipynb
│   ├── 05_hyperparameter_optimization.ipynb
│   └── 06_top3_training_evaluation.ipynb
├── reports/
├── runs/
│   ├── checkpoints/         # saved model weights (.pt)
│   ├── csv_logs/            # per-fold loss logs
│   ├── optuna/              # study.db (SQLite)
│   └── tensorboard/         # TensorBoard event files
└── src/
    ├── __init__.py
    ├── model.py             # MLP architecture
    ├── dataset.py           # build_fold_dataloaders
    ├── train.py             # train_fold, _train_one_epoch, _validate
    └── optuna_objective.py  # make_objective, _suggest_*, _build_*
```

---

## Stack

| Component | Library |
|---|---|
| Neural network | PyTorch |
| Hyperparameter optimisation | Optuna |
| Data splitting / CV | scikit-learn |
| Data manipulation | pandas, NumPy |
| Experiment logging | TensorBoard, CSV |
| Statistical tests | SciPy |

---

## Pipeline Walkthrough

The pipeline is split across six notebooks. Each notebook reads from the outputs of the previous one, so they are designed to be run in order.

### 01 — Data Cleaning

Loads `raw.csv` (1 259 entries, 27 columns). Drops `Timestamp`, `state`, and `comments` as non-informative. Normalises the free-text `Gender` column (48 unique values) into three categories: Male, Female, Other. Clips `Age` to [18, 80] and imputes outliers with the median. Encodes all categorical features (ordinal maps, binary 0/1, unknown-as-No, ternary). Drops 264 rows with missing `work_interfere` (the target), leaving **995 instances**. Saves `cleaned.csv` (for EDA) and `processed.csv` (one-hot encoded, for modelling).

### 02 — Exploratory Data Analysis

Operates on `cleaned.csv`. Analyses each feature type separately — binary (Cramér's V + chi-squared), ordinal (Spearman correlation + Cramér's V), nominal (Cramér's V), and numerical (Kruskal-Wallis). Concludes with a VIF analysis on the candidate feature set. Produces the final list of features kept for modelling.

**Features retained:**

| Feature | Type |
|---|---|
| `treatment` | binary |
| `family_history` | binary |
| `obs_consequence` | binary |
| `benefits` | binary |
| `care_options` | binary |
| `leave` | ordinal |
| `mental_health_consequence` | ordinal |
| `mental_health_interview` | ordinal |
| `Gender_clean_Male` | nominal (OHE) |
| `Gender_clean_Other` | nominal (OHE) |

All VIF values < 5 depict no multicollinearity concern.

### 03 — Preprocessing

Drops irrelevant features identified in EDA. Performs a **stratified 80/20 train/test split** (stratified on `work_interfere`; no random seed, per assignment requirements). Fits `StandardScaler` on the training set only and applies it to both splits. Computes `WeightedRandomSampler` weights from training class frequencies (max/min class ratio: 3.23×). Saves all splits as `.pt` tensors to `data/tensors/`.

### 04 — Model Architecture

Verifies the `MLP` class, `build_fold_dataloaders`, and `train_fold` in isolation. Runs a 10-epoch smoke test on a single fold to confirm the full training loop functions correctly end-to-end before committing to the full optimisation run.

### 05 — Hyperparameter Optimisation

Creates an Optuna study (stored in `runs/optuna/study.db`) and runs **50 trials** of 5-fold CV. The objective function has no fixed random seed for fold splitting (per assignment requirements). Each trial logs per-fold losses to TensorBoard under `runs/tensorboard/optuna/trial_{n}/`. The study is persistent — optimisation can be interrupted and resumed without loss of progress.

**Search space:**

| Hyperparameter | Range |
|---|---|
| Number of hidden layers | 1 – 16 |
| Hidden layer sizes | 16, 32, 64, 128, 256 (per layer) |
| Dropout rate | 0.1 – 0.9 |
| Activation | relu, tanh |
| Optimiser | Adam, SGD |
| Learning rate | 1e-5 – 1e-1 (log) |
| Batch size | 16, 32, 64, 128, 256 |

### 06 — Top 3 Training & Evaluation

Loads the completed study and selects the three best trials by mean CV val loss. Retrains each configuration for **200 epochs** in a fixed-seed 5-fold CV scheme (for reproducible statistical comparison). Saves 5 checkpoints per configuration (15 total). For each configuration, assembles a 5-model ensemble on the test set, reports accuracy, weighted F1, confusion matrix, and per-sample epistemic uncertainty (std of softmax probabilities across models). Runs a **Friedman test** across the three configurations; applies Bonferroni-corrected **Wilcoxon post-hoc tests** if significant.

---

## Results

492 Optuna trials completed. Top 3 configurations by mean 5-fold validation loss:

| Rank | Trial | Val Loss | Hidden layers | Batch size | Optimizer | LR |
|---|---|---|---|---|---|---|
| 1 | #361 | 1.1168 | [16, 64, 256, 64] | 64 | — | — |
| 2 | #188 | — | — | — | — | — |
| 3 | — | — | — | — | — | — |

*(Fill in remaining rows from `study.best_trial` output)*

**Test-set performance (ensemble of 5 fold models):**

| Config | Accuracy | Weighted F1 | Mean uncertainty |
|---|---|---|---|
| 0 (Trial #361) | 0.518 | 0.431 | 0.065 |
| 1 (Trial #188) | 0.543 | — | 0.055 |
| 2 | — | — | 0.050 |

**Statistical tests:**

Friedman test across 3 configurations (5-fold per-model accuracy scores):
- χ² = 1.2000, p = 0.5488

The differences between the three ensembles are **not statistically significant** at α = 0.05. Post-hoc Wilcoxon tests were not performed.

---

## Limitations

**Dataset size.** 995 usable instances is small for a four-class classification task with 10 features. This limits the stability of cross-validation estimates and the statistical power of the Friedman test — with only 5 observations per configuration, even meaningful differences are hard to detect.

**Survey self-report bias.** All features are self-reported. Responses to questions about workplace mental health attitudes (e.g. `mental_health_consequence`, `benefits`) reflect the respondent's perception, not objective workplace policy, which adds noise to any learned relationship.

**Generalisation.** The dataset covers tech industry workers who self-selected into an OSMI survey in 2014. The learned model is unlikely to generalise to other industries, geographies, or time periods without retraining.

**Class imbalance.** The `Sometimes` class accounts for ~47% of instances while `Often` accounts for ~14%. Despite `WeightedRandomSampler`, the model may still be biased toward the majority class, which is reflected in the gap between accuracy and weighted F1.

**`work_interfere` only for diagnosed respondents.** The `work_interfere` question in the survey was only shown to respondents who indicated they have a mental health condition. The 264 dropped rows are not missing at random — they are respondents who answered "No" or "Don't know" to having a condition. This selection effect means the model is trained on a non-representative subset.
