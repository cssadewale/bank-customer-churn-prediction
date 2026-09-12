# Bank Customer Churn Fix Notes

## Fixed

- Python 3.12 deployment pinning and compatible NumPy 2/scikit-learn artifact versions.
- Lazy model/scaler loading before prediction.
- Exact model/scaler feature-schema validation.
- Matplotlib is now imported only when a chart is actually requested, so optional plotting initialization cannot block initial Streamlit rendering.
- Added a lightweight readiness marker immediately after page configuration.

## Required files

```text
app.py
requirements.txt
runtime.txt
models/best_gradient_boosting_churn_model.joblib
models/standard_scaler.joblib
```
