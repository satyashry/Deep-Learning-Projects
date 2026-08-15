# Air Temperature Time Series Forecasting

DataMites™ Internship Project — PR-0019 (PRAICP-1003-AirTempTS)

## Business Case

Develop a machine learning model that can forecast the monthly means of air temperature for the next months.

## Project Goal

Build an ML model to forecast the monthly mean air temperature using time series analysis.

## Dataset

**Surface Air Temperature - Monthly Mean**, recorded at Changi Climate Station, Singapore, published by the National Environment Agency (NEA) via data.gov.sg.

- **Coverage:** 1982-01 to 2020-06 (462 monthly records)
- **Frequency:** Monthly


### Features

| Column | Description |
|---|---|
| `month` | Month for which the mean temperature is recorded (YYYY-MM) |
| `mean_temp` | Average temperature for the month (°C) |

## Approach

1. Data loading and cleaning (checked for nulls, duplicates)
2. Exploratory Data Analysis — trend and seasonality visualization
3. Time series decomposition (trend, seasonal, residual)
4. Stationarity check using Augmented Dickey-Fuller (ADF) test
5. ACF/PACF analysis to understand seasonal lag structure
6. Train-test split (last 12 months held out for testing)
7. SARIMA model building — tested multiple (p,d,q)(P,D,Q,12) combinations, selected best based on AIC
8. Model validation on test data
9. Forecasting future monthly mean temperature

## Model

**SARIMA (Seasonal ARIMA)** — chosen due to the clear yearly seasonal pattern (12-month cycle) present in the data.

## Results

- **MAE:** 0.44
- **RMSE:** 0.50
- **MAPE:** 1.55%

Forecast generated for the next 6 months, saved to `forecast_next_6_months.csv`.

## Files

| File | Description |
|---|---|
| `AirTempTS_Forecasting.ipynb` | Main Jupyter notebook with full analysis and model |
| `AirTempTS.csv` | Dataset used for training/testing |
| `forecast_next_6_months.csv` | Model output — forecasted temperatures |

## How to Run

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
jupyter notebook AirTempTS_Forecasting.ipynb
```

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- statsmodels (SARIMAX)
- scikit-learn (evaluation metrics)

========================================================================================================
