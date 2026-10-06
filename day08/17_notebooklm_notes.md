# Day 08 — NotebookLM Notes: ML Evaluation

## Metrics Cheat Sheet

| Metric | Formula | Use When |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | False positives are costly |
| Recall | TP/(TP+FN) | False negatives are costly |
| F1 | 2×P×R/(P+R) | Balance P and R |
| ROC-AUC | Area under ROC | Ranking quality, model comparison |
| MSE | mean((y-ŷ)²) | Regression, penalize large errors |
| RMSE | √MSE | Same units as target |
| MAE | mean(|y-ŷ|) | Robust to outliers |
| R² | 1-SS_res/SS_tot | Regression quality (1.0=perfect) |

## Cross-Validation Rules

- 5-fold CV: more reliable than single split
- StratifiedKFold: preserves class proportions (use for imbalanced)
- Report: mean ± std from CV (gives confidence interval)
- NEVER: tune hyperparameters then evaluate on same fold

## Hyperparameter Tuning

- GridSearchCV: exhaustive, all combinations, CV for each
- RandomizedSearchCV: random subset, faster, good for large spaces
- Always: select best model → evaluate ONCE on held-out test set

## Interview Facts

1. F1 = harmonic mean (punishes models that are good at one but not both)
2. ROC-AUC of 0.5 = random; 1.0 = perfect; <0.5 = worse than random
3. Accuracy is misleading for imbalanced data (predict majority always = high accuracy)
4. Cross-validation: k fits, each test fold is used once
5. Precision-Recall tradeoff: lowering threshold increases recall, decreases precision
6. GridSearchCV automatically uses CV internally (no separate val set needed)
