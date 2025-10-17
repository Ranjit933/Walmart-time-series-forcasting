# Walmart Time Series Forcasting
![Walmart logo](https://github.com/Ranjit933/Walmart-/blob/main/Walmart%20png/AdobeStock_791702624_Preview_Editorial_Use_Only.jpeg)

## Introduction

**Time series modeling** is a powerful analytical approach used to study data points collected over time and forecast their future behavior. In the retail industry, accurate **sales forecasting** is critical for inventory management, **staffing, marketing strategies**, and overall business planning. **Walmart, being one of the largest retail chains, generates massive amounts of weekly sales data across multiple stores.**

By analyzing this historical sales data, **we can uncover patterns such as trends, seasonality, and the influence of external factors like holidays, unemployment rates, and consumer prices**. Understanding these patterns enables Walmart to anticipate demand, optimize supply chains, and make more data-driven business decisions.

In this project, we focus on weekly sales forecasting for different Walmart stores. Using time series modeling techniques such as **Autoregression (AR), Autoregressive Integrated Moving Average (ARIMA), and Seasonal ARIMA (SARIMA)**, **we aim to predict sales for the upcoming 12 weeks**. By comparing these models, we gain insight into how different approaches handle the challenges of seasonality and trends in retail data. The insights and forecasts generated from this analysis can help Walmart enhance its operational efficiency and maintain a competitive edge in the retail market.

## Problem Statement

Walmart, with its extensive network of retail outlets across the country, faces a critical challenge in **managing inventory levels to align with fluctuating customer demand.** Inconsistent demand patterns, influenced by factors such as holidays, economic conditions, consumer behavior, and external variables like temperature and fuel prices, make it difficult to maintain the right balance between supply and demand.

Failure to anticipate these fluctuations can lead to **overstocking**, which increases holding costs, or **understocking**, which results in missed sales opportunities and dissatisfied customers. To address this, Walmart requires a robust **data-driven forecasting** system that leverages historical sales data to uncover patterns, identify seasonal effects, and predict future demand accurately.

By applying **time series forecasting models** such as **AR, ARIMA, and SARIMA** to the weekly sales data from multiple stores, this project aims to generate reliable 12-week sales forecasts. These forecasts will enable Walmart to improve inventory management, reduce costs, and better serve customer demand across its stores.

## Outline

In this demonstration, we will:

* Prepare the data for time series modeling
  
* Forecast sales using the following models:

* Autoregressive (AR)

* Autoregressive integrated moving average (ARIMA)

* Seasonal autoregressive integrated moving average (SARIMA)

We will analyze the performance of these models using root mean squared error (RMSE) and mean absolute percentage error (MAPE).

Please note that while we could evaluate the performance of the forecasting models using any one of RMSE or MAPE, to obtain a more comprehensive assessment of their performance, we will use both measures for each model.

## About Dataset:-

* **Store:-** Store number

* **Date:-**  Week of Sales

* **Weekly_Sales:-** Sales for the given store in that week

* **Holiday_Flag:-**  If it is a holiday week

* **Temperature:-** Temperature on the day of the sale

* **Fuel_Price:-**  Cost of the fuel in the region

* **CPI:-** Consumer Price Index

* **Unemployment:-**  Unemployment Rate

### Import necessary packages 

##### Import 'numpy' and 'Pandas' for wroking with numbers and dataframes
import numpy as np
import pandas as pd

##### Import 'pyplot' from 'matplotlib' and 'seaborn' for visualizations
from matplotlib import pyplot as plt
import seaborn as sns

##### Augmented Dickey-Fuller(ADF) Test
from statsmodels.tsa.stattools import adfuller

##### Import the 'boxcox' method from 'scipy' to implement the Box-Cox transfomation
from scipy.stats import boxcox

##### Import 'plot_acf' from 'statsmodels' to compute and visualization the autocorrelation function (ACF) for the time series 
from statsmodels.graphics.tsaplots import plot_acf

##### Import 'plot_pacf' from 'statsmodels' to compute and visualize the partial autocorrelation function (ACF) for the time series
from statsmodels.graphics.tsaplots import plot_pacf

##### Import 'ARIMA' from 'statsmodels' for building autoregrssive models
from statsmodels.tsa.arima.model import ARIMA

##### Import 'SARIMAX' from 'statsmodels' for building autoregressive models
from statsmodels.tsa.statespace.sarimax import SARIMAX

##### Import 'mean_squared_error' from 'sklearn' for error computations
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

##### Import and execute method for suppressing warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

![Best 5 store](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Best%205%20store.png)

This chart highlights the **five stores whose weekly sales are least correlated with unemployment rates**. A lower correlation means sales remain steadier even when unemployment fluctuates, making these stores more resilient during economic downturns.

* **Store 36** :- Strongest resilience, correlation just above 0.8, showing sales remain stable despite unemployment shifts.
* ***Store 35** :- Moderate resilience, correlation slightly above 0.6.
* **Stores 14, 21, 30** :- Lower correlations (all below 0.4), indicating partial sensitivity but still relatively insulated compared to other stores.

![Worst](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Worest%205%20store.png)
This chart highlights the **five stores whose weekly sales are most negatively correlated with unemployment rates**. A strong negative correlation means that as unemployment rises, weekly sales fall sharply, making these stores highly sensitive to economic downturns.

* **Store 41** :- Strongest negative correlation (close to -0.9), indicating sales are highly vulnerable to rising unemployment.
* **Store 13** :- Second most affected, with correlation around -0.8.
* **Stores 39, 4, 44** :- Moderate negative correlations (between -0.6 and -0.4), still showing clear unemployment sensitivity.

![Seasonal Trend in Walmart Weekly Sales Month-wise](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Year.png)

Yes, Walmart sales show a seasonal trend.

* Sales spike during November–December (holiday season → Thanks giving, Black Friday, Christmas).

* Another increase may appear during summer (back-to-school season, July–August).

* Weeks with Holiday_Flag = 1 show significantly higher sales compared to non-holiday weeks.

* Seasonal factors like festivals, promotions, weather, and school vacations explain the trend.

![Weekly Sales vs Temperature](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Weekly_Sales%20by%20temp.png)

* The scatter points are spread widely :- meaning sales vary a lot regardless of temperature.

* The regression line is slightly downward :- weak negative correlation: → As temperature increases, weekly sales tend to decrease very slightly.

* But since the slope is almost flat, temperature has only a minor effect compared to other factors (holidays, unemployment, CPI).

![Weekly Sales VS Consumer Price Index (CPI)](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Weekly_Sales%20by%20cpi.png)

* The scatter points are spread widely :- meaning sales vary a lot regardless of  Consumer Price Index (CPI).

* The regression line is slightly downward :- weak negative correlation: → As  Consumer Price Index (CPI) increases, weekly sales tend to decrease very slightly.

* But since the slope is almost flat,  Consumer Price Index (CPI) has only a minor effect compared to other factors (holidays, unemployment, CPI).

![Correlation (Weekly_Sales vs CPI)](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Top_5_store_by_cpi.png)
This chart highlights the **five stores whose weekly sales are least correlated with the Consumer Price Index (CPI).** A lower correlation means sales are more resilient to inflationary pressures, while higher correlation indicates greater sensitivity.

* **Store 17** :- Lowest correlation (just above 0.2), making it the most CPI‑resilient store in this group.
* **Store 41** :- Moderate resilience, correlation around 0.3–0.4.
* **Store 39** :- Mid‑range correlation, showing partial CPI sensitivity.
* **Store 4** :- Higher correlation (~0.6), moderately affected by CPI.
* **Store 44** :- Highest correlation (~0.75) among this group, still relatively sensitive to inflation.

![Correlation (Weekly_Sales vs CPI](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/worst_5_by_cpi.png)

The analysis reveals that **the Consumer Price Index (CPI)** has a negative impact on Walmart’s weekly sales, although the effect differs across stores. In general, stores located in more price-sensitive regions show stronger **negative correlations, meaning sales decrease as CPI rises**. Conversely, a few stores exhibit weak or positive correlations, suggesting their customers are less affected by changes in CPI. Overall, CPI is an important external factor influencing sales, but its impact is not uniform across all Walmart stores.

* **Store 36** :- Strongest negative correlation (close to -1.0), indicating sales are highly sensitive to CPI increases.
* **Store 14** :- Second most affected, with correlation around -0.8.
* **Store 35** :- Significant negative correlation, showing clear CPI vulnerability.
* **Store 43** :- Moderate negative correlation, still impacted by inflation.
* **Store 30** :- Least negative among this group, but still shows CPI sensitivity.

![Average Weekly Sales](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Top_10_store.png)
* The top-performing stores are those with the highest historical weekly sales averages.

* For example, if Store 20, 4, and 14 appear at the top, you’d write:

Store 20 recorded the highest average weekly sales across the dataset, followed by Store 4 and Store 14. These stores consistently outperformed others, indicating higher customer demand or stronger regional market conditions.

![Average Weekly Sales](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Avg_Weekly_sales_store.png)
* **Store 22** highlighted in green as the **top performer**, achieving the highest average weekly sales.
* **Store 7** highlighted in red as the **lowest performer**, indicating potential for improvement.
  
## Store :- 1

![Store 1 Weekly Sales](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Statinaory%20check.png)

## Stationarity Analysis

**In this part of the demonstration, we will perform testes on the time series data to understand weather it is stationary or not. The autoregression modling required the time series data to be statinory. To test this,we will use the following test:**

* **Augmented Dickey-Fuller(ADF) Test**

![Store 1 Data](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/store_1_train_test.png)
* The data was divided into training (all weeks except the last 12) and testing (last 12 weeks) sets.

* This split allows us to validate the forecasting models by comparing predictions with the actual test data.

### Box-Cox Transformation

The Box-Cox Transformation is used to stabilize the variance of a time series. It involves the application of a power transformation to the time series data. Let's import the boxcox method from scipy to implemented this transformation.

![Box-Cox Transformation Data [lambda = 0]](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Weekly_sales_boxcox_store1.png)
**Left plot (Blue) → Original Weekly Sales data of Store 1**

* Large spikes are visible around November 2010, December 2010, and December 2011.

* The variance is quite high, with sales values fluctuating between very high and relatively low levels.

**Right plot (Green) → Box-Cox transformed data (λ = 0, i.e., log transform)**

* The variance has been stabilized across the entire time period.

* The spikes around late 2010 and late 2011 are no longer as extreme.

* The transformed series is smoother and more suitable for time series modeling.

**After transformation, the series became more suitable for time series modeling with AR, ARIMA, and SARIMA.**

![ACF store1](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/pcf_store1.png)
* The ACF plot shows a strong spike at lag 1, followed by very small values within the confidence bands.

* This indicates that the series has short-term correlation only, and there is no strong seasonal autocorrelation visible after Box-Cox transformation.

![PACF Store1](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/pacf_store1.png)
* The PACF plot also shows a clear and significant spike at lag 1, while the remaining lags quickly drop within the confidence interval.

* This suggests that an AR(1) component may be appropriate for modeling the series but we go for all three.

### Autoregressive Models

In this part of the demonstration, we will fit autoregressive models to the data and anaylze their performance using RMSE and MAPE values. We will build the following models:

* Autoregressive (AR)

* Autoregressive integrated moving average (ARIMA)

* Seasonal autoregressive integrated moving average (SARIMA)

### Autoregressive (AR) Method

We will begin by fitting a basic autoregressive model to the training data and analyze its performance. We will use the ARIMA method from statsmodels to build the model.

**Note:** The ARIMA method can also be used to implement other autoregressive models.

Let's import the ARIMA method from statsmodels.

![Store1_train_test](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/store_1_train_test.png)
**Blue Line (Train Data) :** Training set covering the initial period (2010–2012 beginning).

**Green Line (Test Data) :** The last 12 weeks of actual observed sales.

**Purple Line (Predictions) :** Forecasted values from the AR model compared against the test data.

* Predictions (Purple) are generally aligned with the actual test data (Green).The model captured the overall trend, though not all fluctuations.

* Large spikes (e.g., end of 2010 and end of 2011 holiday effects) were not captured well by the AR model.This is expected because AR is simple; ARIMA and SARIMA can capture seasonality better.

* The plot confirms that the **Train/Test split and prediction pipeline** are working correctly.

### Autoregressive Integrated Moving Average (ARIMA) Method
We will now a fit an autoregressive integrated moving average model to the training data and analyze its performance. We will use the ARIMA method from statsmodels to build the model.

Note: The ARIMA method can also be used to implement other autoregressive models.

The parameter of interest in the ARIMA method is the order parameter. It is a 3-tuple of the form *(p,d,q)* with the default value as **(0,0,0)**

For the ARIMA method, we will specify all the values in this tuple. The first and the third values are the *p* and *q*  values or the lag orders obtained from the PACF and the ACF plots respectively. The second value in the tuple is or the differencing order which we shall set as **1**.

**Note:** In ARIMA model, the differencing in already integrated, so we will use df_boxcox instead of df_boxcox_diff.

![Arima Store1](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Arima_store1.png)
**Blue Line (Train Data):** Historical weekly sales from Jan 2010 to early 2012 used to train the model.

**Green Line (Test Data):** Final 12 weeks of actual observed sales held out for validation.

**Purple Line (Predictions):** Forecasted values from the ARIMA model compared against the test set.

* The purple line (Predictions) aligns well with the green line (Test Data), showing that ARIMA captured the general trend.

* Large spikes (e.g., end of 2010 and end of 2011 holiday effects) were not captured well by the AR model.This is expected because AR is simple; ARIMA and SARIMA can capture seasonality better.

* The plot confirms that the **Train/Test split and prediction pipeline** are working correctly.

### Seasonal Autoregressive Integrated Moving Average (SARIMA) Method
We will now a fit a seasonal autoregressive integrated moving average model to the training data and analyze its performance. We will use the SARIMAX method from statsmodels to build the model.

Let's import the SARIMAX method from statsmodels.
![Sarima Store1](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Sarima_store1.png)
#### Conclusion:- Store-1

In this project, predictive modeling techniques were applied to forecast weekly sales for Walmart stores over a 12-week horizon. For demonstration and validation, Store 1 was selected as a case study to compare the performance of three time series models: **Autoregressive (AR), ARIMA, and SARIMA.**
All three models produced nearly identical RMSE values on the test set:

* AR: 1551678.27
* ARIMA: 1551,64.02
* SARIMA: 1551664.02
  
Despite SARIMA’s ability to model seasonality and ARIMA’s differencing capabilities, neither offered a meaningful improvement over the simpler AR model. This suggests that Store 1’s sales pattern is relatively stable, with no strong seasonal fluctuations or complex trends. The AR model, with minimal tuning, delivered comparable accuracy—making it a strong candidate for baseline forecasting.
### Recommendations for Store 1

**1.Adopt AR as the Preferred Model**

* Since AR, ARIMA, and SARIMA all produced nearly identical RMSE values, the simpler AR model should be preferred due to its lower complexity and ease of implementation.

**2.Monitor Seasonal or Event-driven Spikes**

* Store 1’s sales pattern appears stable, but special events such as holiday promotions may still cause sudden spikes. These should be monitored separately to refine forecasts.

**3.Extend to Other Stores**

* Similar analysis should be performed across other Walmart stores to determine whether this stable pattern is unique to Store 1 or consistent across multiple locations.

**4.Use AR as a Baseline for Future Models**

* While AR works well here, it can also serve as a baseline model against which more advanced approaches (like Prophet or ML-based forecasting) can be tested in future work.

## Store :- 2

![Store2 statinary](https://github.com/Ranjit933/Walmart-time-series-forcasting/blob/main/Walmart%20png/Store_2_statinoary.png)
## Stationarity Analysis

**In this part of the demonstration, we will perform testes on the time series data to understand weather it is stationary or not. The autoregression modling required the time series data to be statinory. To test this,we will use the following test:**

* **Augmented Dickey-Fuller(ADF) Test**
