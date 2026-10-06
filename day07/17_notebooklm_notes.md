# Day 07 — NotebookLM Notes: ML Fundamentals

## Algorithm Quick Reference

| Algorithm | Type | Use When | Pros | Cons |
|-----------|------|----------|------|------|
| Linear Regression | Regression | Continuous target, linear relationship | Interpretable, fast | Assumes linearity |
| Logistic Regression | Classification | Binary/multi-class | Interpretable, probabilistic | Linear boundary |
| Decision Tree | Both | Interpretable, mixed features | Interpretable, no scaling | Overfits easily |
| Random Forest | Both | Most problems | Robust, feature importance | Slow to predict |
| KNN | Both | Small dataset, embedding similarity | Simple, no training | Slow at predict time |
| K-Means | Clustering | Find groups | Simple, scalable | Need to specify K |

## Workflow
1. Baseline (predict mean or majority class)
2. Simple model (LogReg / LinReg)
3. Random Forest
4. Evaluate all on val set
5. Pick best, test ONCE on test set

## Interview Facts
1. Random Forest = many trees on random subsets → reduces variance
2. Decision tree depth=20 overfits (train acc 1.0, test acc 0.7)
3. KNN with cosine distance = works well for embedding-based classification
4. K-Means: choose K with elbow method (plot inertia vs K)
5. Feature importance from Random Forest = which features predict target
6. Pipeline(scaler, model) = prevents data leakage in cross-validation

## Common Mistakes
- Not scaling features before KNN or Logistic Regression
- Not setting random_state → non-reproducible results
- Using decision tree without max_depth → always overfits
- Evaluating on test set during model selection → biased results
- Forgetting the baseline → don't know if your model is actually good
