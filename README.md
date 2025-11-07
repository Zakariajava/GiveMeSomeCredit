# Give Me Some Credit — Default Risk Prediction

This project builds a clean, reproducible baseline model to predict **`SeriousDlqin2yrs`** (probability of serious delinquency within 2 years) on the **Give Me Some Credit** dataset using **PyTorch**.

## Project Structure

- `data/`
  - `dataset.csv` — original training data.
- `notebooks/`
  - End-to-end exploratory analysis, preprocessing, model training, and evaluation.
- `models/`
  - Saved model checkpoints (`tabular_mlp.pt`).

## Modeling Pipeline

1. **Data audit & schema**
   - Load `dataset.csv`.
   - Drop technical index column (`Unnamed: 0`).
   - Ensure `SeriousDlqin2yrs` is the last column.
   - Train/Validation/Test split with stratification to preserve class balance.

2. **Exploratory analysis**
   - Univariate distributions and boxplots for all features.
   - Identification of heavy tails, skewness, outliers, and missing values.
   - Focus on:
     - `MonthlyIncome` and `NumberOfDependents` (missingness),
     - highly skewed/long-tailed financial ratios and count features.

3. **Preprocessing**
   - Median imputation for `MonthlyIncome` and `NumberOfDependents`.
   - Missing-value indicators for both.
   - Quantile clipping (1st–99th percentile) for skewed ratios and count variables.
   - Fixed clipping for `age` to [18, 100].
   - `log1p` transform for:
     - `RevolvingUtilizationOfUnsecuredLines`,
     - `DebtRatio`,
     - `MonthlyIncome`.
   - `RobustScaler` for all transformed features.
   - All preprocessing steps are **fit only on the training split** and applied to validation and test via `ColumnTransformer`.

4. **PyTorch data pipeline**
   - Convert preprocessed arrays to tensors.
   - Custom `TabularDataset` and `DataLoader` for train/val/test.
   - Class statistics used to compute:
     - `pos_weight` for positives in `BCEWithLogitsLoss`.

5. **Neural network model**
   - Multi-layer perceptron (MLP) for tabular data:
     - Input dimension = preprocessed features.
     - Hidden layers with `Linear + BatchNorm1d + GELU + Dropout`.
     - Final `Linear` layer outputs a single logit.
   - Kaiming initialization for linear layers.
   - Loss: `BCEWithLogitsLoss(pos_weight=...)` to handle class imbalance.
   - Optimizer: `AdamW` with weight decay.

6. **Training procedure**
   - Mini-batch training with `DataLoader`.
   - mixed-precision (AMP) on CUDA.
   - Validation at each epoch:
     - Track loss, ROC-AUC, Average Precision, Accuracy, F1.
   - Early stopping on **validation ROC-AUC** with patience.
   - Best model checkpoint saved to `models/tabular_mlp.pt`.

7. **Evaluation**
   - Load best checkpoint and evaluate on the held-out test set.
   - Report:
     - Test loss, ROC-AUC, Average Precision, Accuracy, F1.
     - Confusion matrix at threshold 0.5.
     - ROC curve and Precision–Recall curve.
   - Metrics and curves highlight ranking quality under strong class imbalance and illustrate the trade-off between recall and precision for default risk decisions.
