# Pollutant Level Forecasting with XGBoost & Recursive Models 

This project is an introductory exploration of machine learning and time series forecasting models. This repository contains implementations and comparisons of both single-step and recursive multi-step models using the XGBoost algorithm as well as the `skforecast` library.

The original dataset was obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data).

# Project Overview

- Develop baseline single-step forecast model
- Use that model to create a recursive multi-step forecaster to predict farther into the future
- Utilize the `skforecast` library to construct a similar recursive multi-step forecast model and compare results
- Evaluate these models using standard error and loss metrics

# Dataset

The original dataset is composed of hourly pollutant concentration measurements taken from 12 monitoring stations in and around Beijing. The data spanned 4 years from 2013 to 2017. These 12 csv's were grouped into one Dataframe for EDA and the modeling dataset took the mean aggregate of the values over the 12 stations. 

Missing data was sparse and imputed to maintain the frequency of the time series. All numerical data was linearly interpolated, and missing categorical data was filled with the `.ffill` method from `pandas`.

Other preprocessing and feature engineering included the creation of lagged variables containing pollutant concentrations from previous time steps as well as rolling averages. Exogenous variables were handled and included when appropriate, as for recursive forecasting the values of these exogenous variables would be unknown to the model and as such would pose an unrealistic situation.
As I did not have access to seperate forecasts for these variables from the time of prediction, they were left out of those models.

| Model                            | Type       | Horizon  | Notes                                     |
| -------------------------------- | ---------- | -------- | ----------------------------------------- |
| XGBoost (Single-Step)            | Regression | 1 hour   | Predicts next value using lagged features and exogenous features |
| XGBoost (Recursive Multi-step)   | Recursive  | 12 hours | Reuses prior predictions as input         |
| XGBoost (Recursive Multi-step)   | Recursive  | 24 hours | Longer horizon version                    |
| Skforecast `ForecasterRecursive` | Recursive  | 12 hours | Simplified interface for autoregression   |

# Evaluation

Over the scope of this project, the main focus was on PM2.5 levels as the variable of interest, in the future I'd like to expand the scope to include modeling and forecasting of all the pollutants present in the dataset.

Both Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) were used as the primary metrics for this project as they are easily interpretable. The recursive XGBoost model forecasted in rolling windows over the test set and the metrics shown are the mean aggregate over those windows. The recursive model was evaluated on a range of forecast horizons from 1 hour to 24 hours.

# Results

| Model                  | Horizon | MAE  | RMSE |
| ---------------------- | ------- | ---- | ---- |
| XGBoost (Single-Step)  | 1h      | 6.169 | 11.199 |
| XGBoost (Recursive)    | 12h     | 76.129 | 80.544 |
| XGBoost (Recursive)    | 24h     | 75.131 | 81.910 |
| Skforecast (Recursive) | 12h     | 47.319 | 80.041 |

It is important to note that error in recursive models will build up with every prediction over the forecast horizon and so we see a substantial increase in error when moving to the longer distance models. Also relevant is that the single-step model had access to exogenous data and that provides a significant boost to its predictive power.

![image](https://github.com/user-attachments/assets/01a244b5-f059-4c39-8e29-cea30b385048)

The significant increase in error can be clearly seen as the forecast horizon grows.

![image](https://github.com/user-attachments/assets/bd84dbe4-da88-444e-8b01-dae53d7cf005)

# Future Work
- Add a linear regression model and/or some type of naive predictor as another baseline to compare performance
- Expand to include deep learning algorithms like LSTM or TFT
- Incorporate statistical models like SARIMA or SARIMAX as a non-ML approach
- More processing of the data e.g. differencing
- Explore more metrics and model interpretability, like feature importance and SHAP
- Expand modeling capabilities to all 6 pollutants included in the dataset as well as frequency options, e.g. daily, weekly, etc.

