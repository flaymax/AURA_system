# AURA


**AURA** — *AI-powered Unified Reasoning Architecture*.

<div align="center">
  
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)
[![AutoML](https://img.shields.io/badge/AutoML-End--to--End-blueviolet)](https://github.com/flaymax/AURA_system/tree/main/pipeline)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/doc/)
[![Interpretability](https://img.shields.io/badge/Model-Interpretable-success)](https://github.com/flaymax/AURA_system/tree/main/visual)
[![Monitoring](https://img.shields.io/badge/Pipeline-Monitored-informational)](https://github.com/flaymax/AURA_system/tree/main/visual)

[![Linux](https://img.shields.io/badge/Linux-passing-brightgreen)](https://github.com/flaymax/AURA_system/tree/main/)
[![macOS](https://img.shields.io/badge/macOS-passing-brightgreen)](https://github.com/flaymax/AURA_system/tree/main/)
[![Windows](https://img.shields.io/badge/Windows-passing-brightgreen)](https://github.com/flaymax/AURA_system/tree/main/)
</div>

An automated AutoML system that turns raw features into stable, interpretable, and production-ready models using WoE-based feature engineering, statistical selection, and fully monitored pipelines.

Pipeline stages include:

- Automated feature preprocessing with data quality checks and transformations
- WoE-based feature engineering with built-in monotonicity and stability validation
- Statistical feature selection, including stepwise procedures, p-value analysis, and significance testing
- Correlation-aware feature clustering to eliminate redundancy and improve model robustness
- Model training and evaluation with explainability 

Roadmap for core code  (New Components)

| Done | Stage | Status | Notes |
|------|-------|--------|-------|
| ⬜ | `PipelineStage` base class | NEW | Abstract base with fit/transform contract |
| ⬜ | `ScorecardPipeline` orchestrator | NEW | Chains stages, manages state |
| ⬜ | `DataCleaner` | NEW | Basic cleaning logic |
| ⬜ | `TypeDetector` | WRAP | Wraps my `detect_feature_type` |
| ⬜ | `MissingValueHandler` | NEW | Imputation strategies |
| ⬜ | `WoEBinnerStage` | WRAP | Adapter for my `Binner` |
| ⬜ | `PostProcessor` | NEW | Validation after binning |
| ⬜ | `ClusteringStage` | WRAP | Adapter for my `ClusterAnalysis` |
| ⬜ | `StepwiseSelectionStage` | WRAP | Adapter for `iterative_logistic_selection` |
| ⬜ | `FinalFilterStage` | NEW | p-value + VIF filtering |
| ⬜ | `ModelTrainer` | NEW | Final LR training with diagnostics |
| ⬜ | `StabilityAnalyzer` | WRAP | Wraps my PSI code |
| ⬜ | `ReportGenerator` | NEW | Dashboard generation |


