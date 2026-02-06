# models/

Post-training model evaluation stuff. Use this after you have a trained model and want to understand how it behaves.

## Files

### base.py
Core metrics calculation. `PerformanceMetrics` class computes AUC, Gini, KS from predictions. Handles edge cases (all goods, all bads). Also has PSI/CSI calculation for distribution drift.

### stability.py
Time-based monitoring. `StabilityAnalyzer` tracks how your model degrades over time:
- Gini/AUC by period
- PSI vs baseline
- Alerts when things go bad

`ScoreDistributionMonitor` - lightweight PSI checker you can run in production.

### segments.py
Slice-and-dice analysis. `SegmentAnalyzer` breaks down performance by subgroups (age, region, product, whatever). Finds where the model underperforms. `CrossSegmentAnalyzer` does 2D breakdowns.

### comparison.py
A/B testing for models. `ModelComparator` compares challenger vs champion:
- Head-to-head metrics
- Rank correlation (do they agree on ordering?)
- Swap analysis (who would you approve/reject differently?)
- Statistical significance via DeLong test

## Typical usage

```python
from models import StabilityAnalyzer, SegmentAnalyzer

# check stability
analyzer = StabilityAnalyzer(config)
analyzer.evaluate(data, predictions)
print(analyzer.get_alerts())

# check segments
seg = SegmentAnalyzer(config)
seg.evaluate(data, predictions, segment_column='region')
print(seg.get_underperforming_segments())
```
