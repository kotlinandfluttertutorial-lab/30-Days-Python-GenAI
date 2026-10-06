# Day 08 — Day Summary: ML Evaluation

## What You Covered
All classification metrics (accuracy, precision, recall, F1, ROC-AUC), regression metrics (MSE, RMSE, MAE, R²), confusion matrix, cross-validation, hyperparameter tuning with GridSearchCV.

## Key Takeaways
1. Choose your metric BEFORE training based on the business cost of errors
2. CV mean ± std is more reliable than a single train/test split
3. GridSearchCV handles both hyperparameter search AND cross-validation
4. For imbalanced classes: F1 or ROC-AUC, not accuracy
5. The test set is touched exactly once: at the very end

## Tomorrow: Day 09 — End-to-End Machine Learning
Full pipeline: data → preprocessing → training → evaluation → serialization → FastAPI → Docker.
This is Project 1: ML Prediction System.
