# utils/

Helper functions. Grab bag of useful stuff.

## Files

### helpers.py

**`calc_gini_2(y_true, y_pred)`**
Gini coefficient via Lorenz curves. Normalized by the "perfect model" Gini.

Formula: `G_pred / G_true` where G is the area between the Lorenz curve and the diagonal.

Also has some internal helpers:
- `_calc_r2()` - R-squared for single-factor WoE model
- `_delete_duplicates()` - removes duplicate values from list
- `read_file()` - generic file reader

### psi_by_timeperiod.py

**`calculate_psi(expected, actual, buckettype='bins', buckets=10)`**
Population Stability Index. Measures distribution shift between two samples.

PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)

Buckettype options:
- `'bins'` - equal-width bins based on expected distribution range
- `'quantiles'` - percentile-based bins

Interpretation:
- PSI < 0.1: stable
- 0.1-0.25: some shift, investigate
- PSI > 0.25: significant shift, model may need retraining

**`check_psi_feature(df)`**
Batch PSI calculation across months. Expects columns: `month_year`, `sample_type`, `is_pure`.

## Usage

```python
from utils import calculate_psi, calc_gini_2

# check drift
psi = calculate_psi(train_scores, test_scores, buckets=10)
print(f"PSI: {psi:.3f}")

# calculate gini
gini = calc_gini_2(y_true, y_pred)
```
