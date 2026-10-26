# Machine Learning Stock Movement Predictor

Build a simple machine-learning model that predicts whether a stock's closing
price will go up or down on the next trading day. This project is intended for
learning feature engineering, classification, and model evaluation with
historical stock data.

## Project overview

The model will use historical price and trading-volume data to derive features
such as:

- Moving averages
- Trading volume
- Price volatility
- Previous-day returns

The target is a binary label indicating whether the next day's closing price is
higher than the current day's closing price. The workflow will split data into
training and test sets, train a classifier, and evaluate its predictions on
held-out data.

## Implementation plan

Work through these steps in order. Start with one stock and a CSV file so the
data and modeling workflow stay easy to inspect.

### 1. Define the prediction task

- Pick one ticker and a date range for the first experiment.
- Use daily historical data with at least `Date`, `Close`, and `Volume` columns.
   OHLCV data (`Open`, `High`, `Low`, `Close`, and `Volume`) is also suitable.
- For each trading day $t$, use information available at the end of that day to
   predict the direction of the next trading day's closing price.
- Define the target as `1` when the next close is greater than the current close,
   otherwise `0`. The target for the final row is unknown and must be dropped.

### 2. Organize the project and data

- Keep downloaded or original files under `data/raw/`; do not edit the originals.
- Put reusable data-loading, feature-building, and evaluation code in `src/`.
- Keep automated checks in `tests/` and generated charts or reports under
   `outputs/`.
- Record the ticker, date range, data source, and any cleaning decisions so an
   experiment can be repeated.

One possible layout is:

```text
data/
   raw/
src/
   data.py
   features.py
   train.py
   evaluate.py
tests/
outputs/
```

### 3. Load and validate the historical data

- Read the CSV with pandas and parse `Date` as a date/time value.
- Sort rows in ascending date order and remove duplicate dates.
- Check for missing dates, missing values, non-numeric prices, and zero or
   negative prices. Do not silently fill values without documenting the choice.
- Confirm the data has enough history for the longest rolling window you plan
   to use.

### 4. Create the prediction target and features

Create features using only the current day and earlier days. Example features:

- Returns over 1, 5, and 10 trading days, such as `Close.pct_change(5)`.
- 5-day and 20-day moving averages, optionally represented as ratios to the
   current close (for example, `Close / MA_20`).
- Rolling volatility, such as the standard deviation of daily returns over 5 or
   20 days.
- Daily volume and volume relative to a rolling average.
- A target based on the next day's close, for example,
   `Close.shift(-1) > Close`.

Rolling features and returns create rows without complete history. Drop rows
with unavailable feature or target values after constructing all columns.
Check a few dates by hand to verify that the target really refers to the next
trading day. Never include the next day's price or any feature derived from it
among the model inputs; that would leak the answer into training.

### 5. Split data in chronological order

- Use the earliest portion (for example, 80%) for training and the latest
   portion for testing.
- Do not randomly shuffle time-series rows: a random split can let the model
   train on future observations and test on the past.
- Keep the test period untouched until feature choices and model settings are
   settled. If tuning is needed, use a separate, earlier validation period or a
   time-series cross-validation strategy.
- If preprocessing such as scaling is added, fit it on training data only, then
   apply the fitted transformation to validation and test data.

### 6. Train a simple baseline and model

- First measure a baseline, such as always predicting the most common training
   class. This shows whether the model beats a trivial strategy.
- Start with scikit-learn's `LogisticRegression` as a simple classifier.
- Optionally compare it with a `RandomForestClassifier` after the first end-to-
   end run works.
- Train on the feature columns and target from the training period only. Keep
   the target and date columns out of the input features.

### 7. Evaluate on held-out dates

Report more than accuracy, since the up and down classes may be imbalanced:

- Accuracy and balanced accuracy
- Precision, recall, and F1 score for each class
- Confusion matrix
- Baseline score for comparison

Inspect misclassified examples and performance across different time periods.
Do not repeatedly adjust the model based on the final test period; doing so
turns the test set into training feedback.

### 8. Make the result reproducible and usable

- Put the workflow behind a command or `main()` function so it can be rerun for
   a chosen ticker and CSV file.
- Print the data range, feature list, split dates, baseline score, and model
   metrics with each run.
- Save charts or reports to `outputs/` when useful. Avoid saving large generated
   or private datasets to version control.
- Add tests for target alignment, feature calculations, chronological splitting,
   and handling of missing data.

### 9. Suggested milestones

1. Load, sort, and validate one CSV.
2. Build and test the next-day target.
3. Add return, moving-average, volume, and volatility features.
4. Split chronologically and establish a baseline.
5. Train logistic regression and report test metrics.
6. Add tests, document the run steps, and optionally compare another model.

## Tools and concepts

- Python and scikit-learn
- Feature engineering with pandas
- Train/test split and classification
- Model evaluation, including accuracy and confusion matrix

## Important note

This is an educational project, not financial advice. Stock prices are uncertain;
a model's historical performance does not guarantee future results. Avoid using
predictions from this project to make real investment decisions.

## Setup (Windows PowerShell)

1. Make sure Python 3.10 or newer is installed.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install project and development dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```

## Current starter app

```powershell
python -m src.main
```

## Test and lint

```powershell
python -m pytest
ruff check .
```
