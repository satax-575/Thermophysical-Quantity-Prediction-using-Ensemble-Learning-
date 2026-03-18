# Thermophysical Melting Point Prediction — `_VER3_Thermophysical_MP.ipynb`

## Problem Statement

Predict the **melting point (Tm, in °C or K)** of organic molecules from their SMILES string representation. This is a standard cheminformatics regression problem where the goal is to minimise **Mean Absolute Error (MAE)** on held-out data.

---

## Dataset

| Column   | Description                                       |
|----------|---------------------------------------------------|
| `id`     | Unique molecule identifier                        |
| `SMILES` | Canonical SMILES string of the molecule           |
| `Tm`     | Melting point — continuous regression target      |

The dataset also contains pre-computed fingerprint/descriptor columns (the sparse feature set) alongside the raw SMILES.

---

## Repository Structure

```
_VER3_Thermophysical_MP.ipynb   # Main notebook (this file)
train.csv                       # Input dataset
```

---

## Pipeline Overview

```
SMILES
  │
  ├─── [RDKit] ──► ~200 Molecular Descriptors (dense)
  │                       │
  │              VarianceThreshold (drop zero-var)
  │                       │
  │              Mutual Info Regression (MI > 0.01)
  │                       │
  │              LightGBM Importance Pruning
  │                  (gain_pct < 0.1% AND split_pct < 0.1%)
  │                       │
  │                  x_dense  ◄──── Final dense features
  │
  └─── [Raw CSV] ──► Sparse Fingerprint/Descriptor Columns
                          │
                  VarianceThreshold (zero-var)
                          │
                  Sparsity Filter (non-zero fraction ≥ 1%)
                          │
                  Mutual Info Regression (MI > 0)
                          │
                      x_sparse  ◄──── Final sparse features
                          │
          x_final = concat(x_dense, x_sparse)
                          │
              LightGBM Regression + Multi-Seed CV
```

---

## Step-by-Step Details

### 1. Exploratory Data Analysis
- Raw distribution of `Tm` plotted (histogram + KDE).
- Log-transformed distribution (`log1p(Tm)`) also inspected.
- RDKit molecule object explored: atoms, atomic numbers, bond types.

---

### 2. Dense Feature Engineering (RDKit)

```python
from rdkit.Chem import Descriptors

def rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [desc[1](mol) for desc in Descriptors._descList]
```

- Applies the **full RDKit descriptor list** (`Descriptors._descList`) to each SMILES.
- Produces ~200 physicochemical descriptors per molecule (e.g., MolWt, LogP, TPSA, ring counts, hydrogen bond donors/acceptors, etc.).
- Invalid SMILES return `None` and are handled accordingly.

---

### 3. Feature Selection — Dense Features

#### Step A: Variance Threshold
```python
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.0)
df_vt = pd.DataFrame(vt.fit_transform(desc_df), columns=desc_df.columns[vt.get_support()])
```
- Removes **constant features** (zero variance across all molecules).
- These carry no predictive information by definition.

#### Step B: Mutual Information Regression
```python
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(df_vt, target, random_state=11)
mi_scores = pd.Series(mi, index=df_vt.columns).sort_values(ascending=False)
decent_features = mi_scores[mi_scores > 0.01]
```
- Computes **non-parametric mutual information** between each descriptor and `Tm`.
- Retains only features with MI score **> 0.01** — a soft threshold that removes near-irrelevant descriptors while keeping weakly informative ones.
- MI captures **non-linear dependencies**, unlike Pearson correlation.

#### Step C: LightGBM Importance Pruning
```python
lgbm = LGBMRegressor(n_estimators=2000, learning_rate=0.03, random_state=11)
lgbm.fit(dec_fea_df, target)

df_imp["gain_pct"]  = df_imp["Gain"]  / df_imp["Gain"].sum()
df_imp["split_pct"] = df_imp["Split"] / df_imp["Split"].sum()

drop_features = df_imp[
    (df_imp["gain_pct"]  < 0.001) &
    (df_imp["split_pct"] < 0.001)
]["Feature"]
```
- Trains a preliminary LightGBM model on the MI-filtered features.
- Computes **two importance metrics per feature**:
  - **Gain importance**: Total reduction in loss (MAE) contributed across all splits using that feature. Better reflects actual predictive value.
  - **Split importance**: Number of times a feature is used as a split node. May over-count uninformative features used for variance reduction.
- A feature is **dropped only if it ranks below 0.1% on both metrics simultaneously** (AND condition) — conservative pruning that avoids removing features useful under only one criterion.
- Result: `dec_fea_ff` (dense, pruned).

---

### 4. Sparse Feature Engineering

```python
x_sparse = df2.drop(columns=['id', 'SMILES', 'Tm']).copy()
```
- The raw CSV also contains pre-computed **fingerprint/descriptor columns** (likely Morgan fingerprints, MACCS keys, or similar bit-vector representations).
- These are highly sparse (most values are 0).

#### Sparse Feature Selection

**Step A — Variance Threshold:**
```python
vt2 = VarianceThreshold(threshold=0.0)
x_sparse_clean = pd.DataFrame(vt2.fit_transform(x_sparse), ...)
```
Removes columns that are entirely zero (constant).

**Step B — Sparsity Filter:**
```python
non_zero_frac = (x_sparse_clean != 0).mean().sort_values(ascending=False)
x_sparse_clean = x_sparse_clean.loc[:, non_zero_frac >= 0.01]
```
- Keeps only features that are **non-zero in at least 1% of molecules**.
- Extremely sparse features (present in <1% of molecules) add noise and sparsity overhead without meaningful signal in tree models.

**Step C — Mutual Information:**
```python
mi_scores = mutual_info_regression(X=x_sparse_clean, y=target,
                                    discrete_features=False,
                                    n_neighbors=5, random_state=11)
mi_series = mi_series[mi_series > 0]
x_mi = x_sparse_clean[mi_series.index]
```
- MI computed with `n_neighbors=5` (KNN-based estimator).
- Drops features with zero mutual information — these are statistically independent of Tm.

---

### 5. Feature Fusion

```python
x_final = pd.concat([x_dense, x_sparse], axis=1)
```
- Combines the **pruned dense RDKit descriptors** (`x_dense`) with the **filtered sparse fingerprints** (`x_mi`) into a single feature matrix.
- This is the final input to the LightGBM model.

---

### 6. Modelling — LightGBM Regression

#### Baseline Model Parameters

| Parameter           | Value         | Notes                                      |
|---------------------|---------------|--------------------------------------------|
| `learning_rate`     | 0.03          | Low LR → slower convergence, better generalization |
| `num_leaves`        | 48–64         | Controls tree complexity; moderate depth    |
| `max_depth`         | -1            | Unlimited depth (controlled via `num_leaves`) |
| `min_child_samples` | 20–50         | Minimum samples per leaf; regularises small splits |
| `subsample`         | 0.8           | 80% row sampling per tree (stochastic gradient boosting) |
| `colsample_bytree`  | 0.8           | 80% feature sampling per tree               |
| `reg_alpha`         | 0.0 → 1.0     | L1 regularisation (feature sparsity)        |
| `reg_lambda`        | 0.0 → 5.0     | L2 regularisation (weight shrinkage)        |
| `objective`         | `regression`  | Mean Squared Error objective internally     |
| `metric`            | `mae`         | Evaluation metric: Mean Absolute Error      |
| `n_estimators`      | 2000–3000     | Number of boosting rounds                   |
| `random_state`      | 11            | Reproducibility seed                        |
| `force_col_wise`    | True          | Memory-efficient column-wise histogram      |

> **Note:** Two parameter sets appear in the notebook:
> - **Exploration set** (`reg_alpha=0.0`, `reg_lambda=0.0`, `num_leaves=64`): minimal regularisation for SHAP analysis.
> - **CV/tuning set** (`reg_alpha=1.0`, `reg_lambda=5.0`, `num_leaves=48`, `min_child_samples=50`): stronger regularisation for generalisation.

---

### 7. Cross-Validation Strategy

```python
N_SPLITS = 5
SEEDS = [42, 101, 777]

for seed in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, val) in enumerate(kf.split(X), 1):
        ...
```

- **5-fold KFold** cross-validation, repeated over **3 random seeds**.
- Total = **15 model fits** per experiment.
- OOF (out-of-fold) predictions are aggregated across all folds.
- Final reported score: **mean MAE across all seeds and folds**.

#### Why multi-seed CV?
- A single KFold split can be lucky/unlucky depending on how data is partitioned.
- Averaging over seeds reduces variance in the CV estimate — gives a more stable and trustworthy MAE estimate.

#### Metric: Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- Measures average absolute deviation of predictions from true Tm.
- Less sensitive to outliers than RMSE.
- Directly interpretable in the same units as Tm (°C or K).
- LightGBM's internal `metric = "mae"` monitors this on each validation set during training (early stopping not shown in visible code but implied by the large `n_estimators`).

---

### 8. Hyperparameter Optimisation (Optuna — Optional)

```python
# import optuna
# def objective(trial):
#     params = {
#         "num_leaves": trial.suggest_int("num_leaves", 20, 150),
#         ...
#     }
#     return run_cv(X, y, params, n_splits=5)
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)
```

- Optuna integration is **present but commented out**.
- Uses **Bayesian optimisation (TPE sampler)** to search hyperparameter space.
- Objective: minimise CV MAE.
- The notebook confirms: **Optuna CV MAE < Baseline CV MAE**, meaning tuning yielded measurable improvement.

---

### 9. SHAP Interpretability Analysis

```python
from shap import TreeExplainer
explainer = TreeExplainer(baseline_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="dot")   # Beeswarm
shap.summary_plot(shap_values, X, plot_type="bar")   # Global importance
```

- **TreeExplainer** computes exact SHAP values for tree-based models — no approximation needed.
- **Dot (beeswarm) plot**: Shows each molecule as a dot; x-axis is SHAP value (impact on Tm prediction), colour encodes feature value magnitude. Reveals both direction and magnitude of each feature's effect.
- **Bar plot**: Mean absolute SHAP values — a clean global feature importance ranking.
- SHAP values are **consistent and locally accurate**: they sum to the model output, unlike gain/split importances.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Two-stage feature selection (MI + LightGBM importance) | MI catches non-linear relevance early; LightGBM importance removes redundant features after fitting |
| AND condition for importance pruning | Conservative — avoids dropping features that matter under one criterion but not the other |
| Multi-seed CV | Reduces variance of the MAE estimate; catches overfitting to a single data split |
| Dual feature streams (dense + sparse) | RDKit descriptors encode interpretable physicochemical properties; fingerprints encode structural patterns — complementary information |
| Sparse feature sparsity filter (≥ 1%) | Prevents near-constant sparse columns from adding noise while saving memory |
| High `n_estimators` (2000–3000) with low `lr=0.03` | Typical LightGBM best practice: more trees + lower learning rate generalises better than fewer trees + high LR |

---

## Dependencies

```
rdkit
lightgbm
scikit-learn
shap
optuna          # optional, for hyperparameter tuning
pandas
numpy
seaborn
matplotlib
```

---

## How to Run

1. Place `train.csv` at `/content/train.csv` (Google Colab path) or update the path in the data loading cells.
2. Install dependencies:
   ```bash
   pip install rdkit lightgbm shap optuna
   ```
3. Run all cells top-to-bottom. Feature selection and CV will execute sequentially.
4. To enable Optuna tuning, uncomment the Optuna cell block and run.

---

## Notes & Caveats

- **Invalid SMILES** return `None` from `Chem.MolFromSmiles()` and need to be handled (dropped or imputed) before training.
- The notebook uses **no early stopping** in CV — `n_estimators` is fixed. Adding early stopping with `eval_set` would reduce overfitting risk on specific folds.
- The **target is not log-transformed** for modelling, despite the EDA showing a skewed distribution. Experimenting with `log1p(Tm)` as the target (and expm1 post-prediction) may improve MAE.
- The sparse fingerprint columns from the CSV are likely **pre-computed** (not generated in-notebook) — their exact type (Morgan, MACCS, etc.) is not specified.
