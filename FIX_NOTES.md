# Bank Customer Churn Fix Notes

## Fixed

- Added Python 3.12 deployment pinning and compatible binary-wheel dependencies.
- Added lazy artifact loading so a missing/incompatible model cannot block the whole dashboard from rendering.
- Added explicit existence checks for both model files.
- Added exact validation of the model's 11 feature names and scaler's five feature names.
- Added visible error handling during prediction-time model loading.

## Required files

These files must remain in the repository's `models/` directory:

```text
models/best_gradient_boosting_churn_model.joblib
models/standard_scaler.joblib
```
