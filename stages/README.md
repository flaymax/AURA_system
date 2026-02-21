# stages pipeline notes

This is the pipeline itself. Every step your data goes through lives here, from raw input to a trained model. Each stage inherits from `PipelineStage` (in `core/base.py`) and follows the same `fit()` / `transform()` / `fit_transform()` interface. Every stage produces a `StageResult` with transformed data, logs (JSON-serializable for the web frontend), and metadata.

Stages run in a fixed order. The output of one feeds directly into the next. If a stage decides to drop a feature, it's gone for everything downstream.





## The pipeline flow

```
raw data
  │
  ▼
preprocessing/cleaner.py        ← throw out garbage columns
  │
  ▼
preprocessing/type_detector.py  ← figure out what each column is
  │
  ▼
preprocessing/reject_inference.py  ← (optional) deal with selection bias
  │
  ▼
transformation/woe_binner.py    ← bin features, compute WoE, filter by IV/PSI/monotonicity
  │
  ▼
selection/clustering.py         ← group correlated features, pick one per group
  │
  ▼
selection/stepwise.py           ← forward-backward selection with early stopping
  │
  ▼
selection/interactions.py       ← (optional) try X1*X2 / X1/X2 / X1-X2 combos
  │
  ▼
selection/final_filter.py       ← last check: p-values, VIF, coefficient signs
  │
  ▼
modeling/trainer.py             ← fit logistic regression, build scoring formula
  │
  ▼
visualization/*                 ← charts for everything above
```

---

## preprocessing/

Gets the data into a usable state before any real modeling happens.

### cleaner.py — `DataCleaner`

Removes features that are obviously useless:
- Too many nulls (>97% by default)
- No variance (>97% same value)
- Non-numeric types (strings, objects)

You can protect specific columns from removal via `keep_columns`. Logs every drop decision with a reason code (`NULL`, `LOW_VAR`, `NON_NUM`) so the frontend can show the user what was removed and why.

### type_detector.py — `TypeDetector`

Classifies each surviving column into a type:
- **Binary**: exactly 2 unique values
- **Categorical**: 3-5 uniques
- **Discrete numeric**: 6-30 uniques
- **Continuous**: >30 uniques
- **ID-like**: unique ratio > 0.9, probably a row identifier

This affects how downstream stages treat the feature (e.g., binning strategy).

### reject_inference.py — `RejectInferenceStage`

Optional. Only matters when your training data has selection bias — you only see outcomes for records that were approved/selected, and need to infer labels for the rejected ones.

Four methods:
- **hard_cutoff**: score below threshold = bad. Simple but crude.
- **fuzzy**: train a model on accepted, predict P(bad) for rejected, assign labels probabilistically. Weights reflect confidence (`|P(bad) - 0.5| * 2`).
- **parceling**: bin by score, use each bin's observed bad rate from accepted data to assign labels to rejected records in the same bin.
- **augmentation**: just slap a fixed bad rate on all rejects. Last resort.

Outputs augmented target + sample weights for downstream stages.

---

## transformation/

Turns raw feature values into something the model can actually learn from.

### woe_binner.py — `WoEBinnerStage`

The heaviest stage in the pipeline. Wraps the legacy `Binner` class from `pipeline/binner.py`.

**What it does**: for each feature, find optimal bin boundaries and compute Weight of Evidence per bin.

**How it bins**: decision tree, not percentiles. Tries tree depths 1 through `power` (default 3, so up to 8 leaf bins). Picks the depth that maximizes IV (or R²). Missing values always get their own bin.

**What it computes per feature**:
- WoE per bin: `ln(good% / bad%)`
- IV (Information Value): `sum((good% - bad%) * WoE)` — this is the main filter. Features with IV below `min_iv` get dropped.
- Gini, HHI, R² as extra diagnostics

**Monotonicity check**: WoE should generally trend in one direction across bins (higher value = consistently more or less risky). The stage detects violations and can:
- `"enforce"`: drop non-monotonic features
- `"warn"`: log it, keep the feature
- `"ignore"`: don't even check

**PSI check**: compares bin distributions between train and test. Catches features where the population shifted between samples. Same three modes (enforce/warn/ignore). Standard thresholds: <0.1 stable, 0.1-0.25 moderate, >0.25 unstable.

After this stage, every feature column contains WoE values instead of the original values.

---

## selection/

Three (optionally four) stages that narrow down features to only the ones worth keeping.

### clustering.py — `ClusteringStage`

Wraps the legacy `ClusterAnalysis` from `pipeline/clustering.py`.

**Problem**: you might have 50 features that survived binning, but half of them are saying the same thing (correlated). Keeping all of them just adds noise.

**Algorithm**: hierarchical clustering with correlation distance (`1 - |corr|`). Features with correlation above the threshold end up in the same cluster. One representative per cluster.

**Selection strategies** (pick the best from each cluster):
- `max_train` / `max_test`: highest Gini on that sample
- `closest_train_test`: smallest gap between train and test Gini (stability pick)
- `center_cluster`: closest to the cluster centroid in correlation space

**P-value filter** (optional, on by default): after picking representatives, runs univariate logistic regression on each. Features with p-value > 0.05 get removed. Catches statistically insignificant ones early before stepwise wastes time on them.

### stepwise.py — `StepwiseSelectionStage`

Wraps `iterative_logistic_selection()` from `src/binary_logistic.py`.

Classic forward-backward stepwise:
1. **Forward**: try adding each remaining feature. Keep the one where the likelihood ratio test says the model improved significantly (p < alpha).
2. **Backward**: check all features currently in the model. If any have a Wald test p-value above alpha, remove the worst one.
3. Repeat until nothing changes.

Tracks AUC on train and test at every step.

**Early stopping**: monitors test AUC as features are added. If it hasn't improved by at least `min_improvement` for `patience` steps in a row, stop. If `restore_best=True`, roll back to the feature set that gave the best test AUC. This is the main overfitting guard in the pipeline.

### interactions.py — `InteractionDetectorStage`

Optional (disabled by default). Tries to find interaction terms that improve the model beyond what individual features give you.

**Interaction types**:
- Multiplicative: `X1 * X2`
- Ratio: `X1 / X2` (safe division, handles zeros)
- Difference: `X1 - X2`

**Process**:
1. Generate candidate pairs. If there are too many (>50), prioritize pairs with moderate correlation (~0.3) — not too correlated (redundant) and not too independent (unlikely to interact meaningfully).
2. For each pair and type, create the interaction column.
3. Test significance via logistic regression p-value.
4. Measure AUC improvement on test set.
5. Keep the top N interactions meeting both p-value and AUC improvement thresholds.

### final_filter.py — `FinalFilterStage`

Last gate before model training. Three checks:

1. **P-value**: fits logistic regression, computes Wald test p-values. Drops features where p > threshold (default 0.05).

2. **VIF (Variance Inflation Factor)**: catches multicollinearity that clustering might have missed. `VIF = 1 / (1 - R²)` where R² comes from regressing each feature on all others. VIF > 5 means the feature is largely explained by its neighbors. Iterative removal: drop worst offender, recalculate, repeat until all are clean.

3. **Coefficient sign check** (optional): verifies that each coefficient has the expected direction. A feature where higher WoE should mean lower risk but has a positive coefficient — that's suspicious.

---

## modeling/

### trainer.py — `ModelTrainerStage`

Fits the final logistic regression. No regularization by default — for interpretability you want unbiased coefficient estimates, and by this point the features have been heavily filtered so overfitting is less of a concern.

**What it computes**:
- AUC, Gini, KS on both train and test
- Coefficient standard errors from the Hessian: `Cov = inv(X'WX)` where `W = diag(p*(1-p))`
- Wald p-values per feature

**Points-based scoring**: converts the logistic model into a formula:
```
Factor = PDO / ln(2)
Score = base_score + Factor * (intercept + sum(coef_i * woe_i))
```
Defaults: base_score=600, PDO=20 (20 points to double the odds). Higher score = lower risk.

**SQL generation**: `to_sql()` outputs a SQL SELECT statement you can deploy straight to a database. No Python runtime needed in production.

**Bootstrap CI** (optional): resamples train data N times (default 1000), refits, reports percentile-based confidence intervals for AUC/Gini/KS. Stratified sampling to preserve class balance.

---

## visualization/

Two backends for every chart: Matplotlib (static, for PDFs and reports) and Plotly (interactive, for web dashboards). Plotly files have a `_pl` suffix.

| Category | Charts | What it shows |
|----------|--------|---------------|
| **performance** | ROC curve, AUC trend, KS plot | Model discrimination power |
| **distribution** | Score histogram, hits by bucket, density comparison | How scores spread across good/bad |
| **gains** | Gains chart, lift chart, capture rate table | How much of the target you catch at each cutoff |
| **rates** | AR/BR curve, AR/BR table, swap analysis | Approval rate vs bad rate tradeoff |

`BaseVisualization` (matplotlib) and `BasePlotlyVisualization` define the interface: `plot()`, `get_data()`, `save()`. Every chart inherits from one of these.

Plotly uses a warm color palette: terracotta, dark blue-gray, sage green, coral red, sandy orange. All charts can export to PNG, SVG, PDF, HTML, or raw JSON.

---

## How stages talk to each other

They don't, directly. Each stage gets a DataFrame in, does its thing, and passes a (possibly smaller) DataFrame out. The `ScorecardPipeline` orchestrator in `core/pipeline.py` handles the wiring.

What flows between stages:
- **Feature columns shrink** at every step. Cleaning drops garbage, binning drops low-IV features, clustering drops correlated duplicates, stepwise picks the best subset, final filter cleans up the rest.
- **WoE transformation** at the binning stage replaces raw values with WoE-encoded values. Everything after binning works on WoE columns.
- **sample_type column** (0=train, 1=test, 2=validation) is preserved throughout and used by multiple stages for train/test splitting.
- **Logs accumulate** — each stage writes its own JSON-serializable log section. The pipeline collects them all for the web frontend.

## Adding a new stage

1. Create a class inheriting from `PipelineStage` (in `core/base.py`)
2. Implement `fit()`, `transform()`, `fit_transform()`, `validate()`
3. Write `_build_logs()` returning a JSON-serializable dict
4. Add it to the subfolder's `__init__.py`
5. Export it from `stages/__init__.py`
6. Wire it into `runner.py` and `core/pipeline.py`
