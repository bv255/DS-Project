# COMP0047 Data Science Project

This repository studies **bull/bear market regime classification and prediction for the S&P 500**.

In practice, this is a **notebook-first research repo**. Most of the project logic lives in Jupyter notebooks, while `src/` contains a small amount of reusable Python code for preparing a cross-validation dataset and training an LSTM classifier.

## What Is In This Repo

The project is organized around four stages:

1. **Data collection**
2. **Feature engineering**
3. **Regime labeling / segmentation**
4. **Regime prediction**

The core idea is:

- collect market and macroeconomic time series
- engineer technical and macro features
- assign bull/bear regime labels using segmentation methods
- train models to predict future regimes at different horizons

## Current Repository Reality

The repository is not a fully packaged pipeline. It is closer to a research workspace with:

- pre-generated CSV datasets in `data/`
- exploratory and modelling notebooks in `notebooks/`
- one LSTM training script in `src/models/lstm.py`
- one helper script in `src/scripts/cv_dataset.py`

## Repository Structure

```text
data/
notebooks/
  01_data_collection.ipynb
  02_feature_engineering.ipynb
  03_regime_segmentation/
    multivariate/
    univariate/
  04_time_lag_features.ipynb
  05_prediction/
src/
```

## Data Files

The shipped CSV files already contain most intermediate outputs, so you do not need to rerun the whole pipeline to inspect results.

### `data/raw_data.csv`

Raw aligned market and macro data. Columns include:

- S&P 500 (`GSPC`)
- VIX
- SPY volume
- Gold
- Oil
- GDP
- Core inflation
- Unemployment
- M2
- Sentiment

### `data/master_data.csv`

Feature-engineered dataset with technical indicators and macro transformations, including:

- returns over multiple horizons
- smoothed returns
- RSI
- MACD features
- drawdown
- VIX changes
- year-over-year macro features

This file also has a `Regime` column, though early rows contain missing values.

### `data/labeled_dataset.csv`

Main rule/segmentation-labelled dataset used by the reusable Python scripts. It contains:

- engineered features
- `regime` labels (`bull` / `bear`)
- a numeric `segment`

### `data/cv.csv`

Prepared modelling dataset containing stationary features plus a binary target. This appears to be an exported modelling table rather than the primary source used by the LSTM script.

### `data/multivariate_gmm_labeled_dataset.csv`

Alternative labelled dataset from the multivariate GMM regime segmentation workflow, including:

- consumer sentiment features
- `prob_bull`
- `prob_bear`

## Notebook Workflow

The notebooks form the main project narrative.

### 1. Data collection

`notebooks/01_data_collection.ipynb`

- pulls market data and macroeconomic series
- mentions using a `FRED_API_KEY` in a local `.env` file if regenerating data
- produces the raw input dataset that is already committed

### 2. Feature engineering

`notebooks/02_feature_engineering.ipynb`

- documents feature definitions in detail
- builds technical indicators and macro transformations
- produces the engineered dataset used downstream

### 3. Regime segmentation / labeling

`notebooks/03_regime_segmentation/`

This folder contains several alternative labeling approaches:

- `univariate/rule_based_regime_segmentation.ipynb`
  - uses rule-based logic based on moving averages
- `univariate/simple_markov_regime_segmentation.ipynb`
  - applies Markov-switching style regime modelling
- `univariate/rule_based_time_horizon_lagged_features.ipynb`
  - tests prediction horizons and constructs lagged targets
- `multivariate/multivariate_changepoint_regime_segmentation.ipynb`
  - uses change point detection with `ruptures`
- `multivariate/multivariate_gmm_regime_segmentation.ipynb`
  - uses Gaussian mixture modelling for regime assignment
- `multivariate/multivariate_hmm_regime_segmentation.ipynb`
  - applies a multivariate HMM approach

This is where the repo experiments with different definitions of “bull” and “bear”.

### 4. Time-lag feature construction

`notebooks/04_time_lag_features.ipynb`

- creates lagged prediction targets
- focuses on forecasting future regimes instead of just labeling the present

### 5. Prediction notebooks

`notebooks/05_prediction/`

These notebooks contain classical ML experiments for different horizons:

- `prediction_basecase_1day.ipynb`
- `prediction_basecase_5days.ipynb`
- `prediction_basecase_20days.ipynb`
- `prediction_crossValidation.ipynb`

From the imports, these notebooks use tools such as:

- logistic / statistical modelling
- XGBoost
- cross-validation and standard classification metrics

## Python Code

### `src/scripts/cv_dataset.py`

This is a small helper that:

- loads `data/labeled_dataset.csv`
- converts `regime` into a binary target
- engineers a few additional modelling features:
  - `Risk_Adj_Return_20d`
  - `Relative_Volume`
  - `MACD_Hist_Accel`
- returns a compact modelling DataFrame of stationary features plus `regime_binary`

### `src/models/lstm.py`

This is the main reusable training script in the repo.

It does the following:

- loads the labelled dataset through `create_cv_dataset()`
- creates future targets for three forecast horizons: `1`, `5`, and `30` days
- uses a `20`-step rolling window to build sequential samples
- splits the data chronologically:
  - first `85%` for cross-validation
  - last `15%` for test
- trains a TensorFlow LSTM classifier with:
  - two LSTM layers
  - batch normalization
  - dropout
  - L2 regularization
- runs `TimeSeriesSplit` cross-validation
- saves:
  - trained `.keras` models
  - CV summary CSVs
  - training-curve plots
  - confusion matrix plots
  - classification reports
  - test metrics

Training output is written under `src/trained/` when the script is run.

## How To Run

### 1. Install dependencies

Use the requirements file:

```bash
pip install -r requirements.txt
```

Note:

- `environment.yaml` is currently empty, so `conda env create -f environment.yaml` will not work as-is
- some notebook workflows may also require Jupyter and a valid FRED API key if you want to regenerate data

### 2. Run the LSTM pipeline

Train and evaluate:

```bash
python3 -m src.models.lstm --train
```

Evaluate using previously saved models:

```bash
python3 -m src.models.lstm
```

If no trained models exist yet, the evaluation-only mode will fail until `--train` has been run.

## Dependency Notes

`requirements.txt` includes libraries used across notebooks and scripts, including:

- pandas / numpy
- scikit-learn
- tensorflow / keras
- matplotlib / seaborn / plotly
- yfinance / fredapi
- statsmodels
- ruptures
- hmmlearn
- xgboost
- shap
- ta