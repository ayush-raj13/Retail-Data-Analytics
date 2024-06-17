# Retail-Data-Analytics

This README file provides a comprehensive guide to the sales forecasting process implemented in the provided Python notebook. The notebook covers various stages including data acquisition, data pre-processing, data visualization, machine learning model building, and performance evaluation.

## Table of Contents
1. [Data Acquisition](#data-acquisition)
2. [Data Pre-processing](#data-pre-processing)
3. [Data Visualization](#data-visualization)
4. [ML Model Building](#ml-model-building)
5. [Performance Evaluation](#performance-evaluation)

## Data Acquisition

### Description
Data acquisition involves loading and importing the necessary datasets for the sales forecasting task. In this notebook, three main datasets are used:
- Stores Data
- Features Data
- Sales Data

### Code
```python
import pandas as pd
import numpy as np

# Load datasets
df_stores = pd.read_csv('../input/retaildataset/stores data-set.csv')
df_features = pd.read_csv('../input/retaildataset/Features data set.csv', parse_dates=['Date'])
df_sales = pd.read_csv('../input/retaildataset/sales data-set.csv', parse_dates=['Date'])
```
## Data Pre-processing
### Description
Data pre-processing involves cleaning and transforming raw data into a suitable format for analysis. This includes handling missing values, converting data types, and merging datasets.

Code
```python
# Display the first few rows of the sales data
df_sales.head()

# Check for missing values in the features data
df_features.isna().sum()

# Plot to visualize missing data in unemployment feature
df_features.Unemployment.plot()
```
### Steps
- Load and inspect each dataset.
- Handle missing values and data inconsistencies.
- Merge datasets based on common keys (e.g., Store, Date).
- Data Visualization
### Description
Data visualization helps in understanding the patterns and relationships in the data. Various plots and charts are used to visualize trends, seasonality, and other important aspects of the data.

Code
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot sales over time
plt.figure(figsize=(14, 7))
plt.plot(df_sales['Date'], df_sales['Weekly_Sales'])
plt.title('Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()

# Visualize the distribution of sales
sns.histplot(df_sales['Weekly_Sales'], bins=50)
plt.title('Distribution of Weekly Sales')
plt.xlabel('Weekly Sales')
plt.ylabel('Frequency')
plt.show()
```
### Visuals
1. Time series plots to visualize sales over time.
2. Distribution plots to understand the distribution of sales values.
## ML Model Building
### Description
This code snippet uses the Holt-Winters exponential smoothing method (ExponentialSmoothing from statsmodels) to forecast future values of weekly sales. It specifies additive trend and seasonal components with a seasonal period of 52 weeks, fits the model to the first 120 data points, and then generates a forecast for the next 34 periods.

Code
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit_model = ExponentialSmoothing(df_by_date_new['Weekly_Sales'][:120],
                                 trend = 'add',
                                 seasonal = 'add',
                                 seasonal_periods = 52).fit()

prediction = fit_model.forecast(34)
prediction
```
## Performance Evaluation
### Description
Performance evaluation involves assessing the accuracy and effectiveness of the model using metric such as Mean Absolute Error (MAE).

Code
```python
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("Mean Absolute Percentage Error = {a}%".format(a=mean_absolute_percentage_error(df_by_date_new.Weekly_Sales[120:],prediction)))
```
### Metrics
- Mean Absolute Error (MAE): Measures the average magnitude of the errors.
