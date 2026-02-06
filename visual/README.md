# visual/

Plotting utilities. Mostly for model diagnostics and monitoring dashboards.

## Files

### gain_charts.py

**`get_cum_percentile_buckets(df, column_to_sort, target_column, baseline_ar)`**
Cumulative gains table. Sorts by score, then shows target rate at each percentile cutoff (5%, 10%, ..., 75%).

Returns DataFrame with buckets + dict for quick lookup.

**`get_cumul_month_percentile_buckets(...)`**
Same thing but broken down by month. Compares two scores side-by-side ("было"/"стало" - before/after). Useful for model replacement analysis.

**`df_style(val)`**
Just returns bold font style. For pandas styling.

### pyplot.py

**`build_weekly_risk_dashboard(data, target, score_main, date_col, ...)`**
Plotly dashboard showing weekly trends:
- Target rate over time
- Model scores (main + up to 2 alternatives)
- Optional confidence bands
- Optional volume bars

Good for weekly monitoring reports.

### visual_woe_diagnostic_monotonik.py

**`woe_monotonicity_diagnostics(data, feature_name, target_name, n_bins=15)`**
Visual check for feature monotonicity. Plots:
- WoE by bucket with confidence intervals
- Linear fit line
- Shows AUC, IV, R² in title

Uses quantile-based binning. Confidence intervals from asymptotic WoE variance: `sqrt(1/events + 1/non_events)`.

## When to use what

- **gain_charts**: approval rate analysis, model comparison at different cutoffs
- **pyplot dashboard**: production monitoring, weekly reviews
- **woe_monotonicity**: feature engineering, checking if WoE relationship makes sense
