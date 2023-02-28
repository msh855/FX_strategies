# README

### 1. PREDICATBILITY OF US EXCESS RETURN USING C-MEAN FORECASTING: UNFINISHED PROJECT LOOKING FOR HELP

#### 1.1 GOAL
- Construct a short term trading model that switches in steps on a monthly basis between S&P500 and Treasuries using a forecasted return for next month. The forecasted return for next month is based on a combined forecast of univariate expanding window regressions of monthly excess returns on several predictors.

- feb 2023 update: unfinished as oos r-squared seem too strange. Haven't found code error/data error yet. If you do, please let me know

variables. Back-test the model on monthly data.
#### 1.2 PROCESS

Response/dependent variable/endog/target variable: 
    - monthly excess returns (S&P500 return - Treasury return) transformed to monthly log returns
- Regressors/independent variables/features/exogs/predictors:
    - market variables: momentum 1m, momentum 3m, momentum 1m, momentum 1m, distance to 1m moving average of the excess returns.
    - seasonality: monthly dummies, presidential cycle?
    - monetary variables: 10-2 term spread,pce deflator, real Fed rate, real 10y yield, real Tbill rate, Baa yield spread
    - monetary variables shocks: 1-month changes in the monetary variables
    - macro variables:  1m change in LEI, ISM_index, 1m change in USA OECD amplitude adjusted, 1m change in consumer sentiment,

- Collect data from csv_file that contains all necessary data (response and predictors). Clean and initial conversions:
    - Create correct **raw time_series formats**: dropna(), set_index, Datetime index with correct frequency, resampled to end of month so that all variables will have aligned dates to concat later on. Squeeze df to series, check for correct data-types, check data-formats <br> 
    (percentage in decimals,...), name the index and the value of the series, check shape and plot. <br>
    - Create a **lagged time_series** if necessary for macroeconomic predictors that are published in a period with reference to a lagged period. <br> 
    Make the series **point-in-time**.<br>
    - Create **log_return series** from simple return series: (np.log(1+R) = np.log1p(R)). <br>
<br>
- Construct and display **descriptive statistics** for all variables (response and predictors, simple returns and log returns). Plot them.

- Test for a trend-stationary process on the raw data (but on log returns). <br>
- Create time_series of z-_scores and trim them to remove outliers to min/max = -3/+3:
    - The z-scores make the time_series easily comparable (also after regression to compare strength of beta effects visually) with <br> the same scale.Create 2 sorts of z_scores per predictor: <br>
        - in_sample_z_scores: z_scores will be used for in-sample predictive regression on the full dataset. <br> 
        - In-sample z-scores create look-ahead bias but regression already creates look-ahead bias. <br>
                For in-sample testing hence, ok. <br>
         - out_of_sample_z_scores: time_series of z-_scores per predictor calculated recursively (expanding window: with a warm-up period and then calculated from start till the next value). The out_of_sample_z_scores will be used for out-of-sample predictive regressions. <br>

- Test for a trend-stationary process on the transformed data. Drop any series that is still stationary for future testing. <br>

- perform OLS in-sample: in-sample means perform univariate ols-regression using the full dataset.

for each variable: <br>

- Run a predictive in-sample regression on the lagged variables (z-scores, 1-month lagged)<br>

$$ r_{t} = C + \beta x_{t-1}+\epsilon_{t} $$

$$ r_{t} = \text{log excess return of month t}  $$
$$ x_{t-1} = \text{predictor value previous month t-1 = predictor lagged 1 month}  $$
$$ C_{t} = \text{Intercept coefficient estimated in sample}  $$
$$ \beta_{t} = \text{beta coefficient estimated in sample}  $$

for each variable and for each time step t: <br>

- Run a predictive expanding regression on the lagged variables (expanding z_scores of the variables, warmed up n = 60 months) with a warm-up of n = 60 months of the form:<br>

$$ r_{t} = C_{t} + \beta_{t}x_{t-1}+\epsilon_{t} $$

$$ r_{t} = \text{log excess return of month t}  $$
$$ x_{t-1} = \text{predictor value previous month t-1 = predictor lagged 1 month}  $$
$$ C_{t} = \text{Constant coefficient estimated at month t}  $$
$$ \beta_{t} = \text{beta coefficient estimated at month t}  $$

- Each month - and for each predictor - we have estimated coefficients at month t that used all data from the warm-up date until month t. In contrast to in-sample predictive regressions our coefficients are now time-varying and are updated as new information becomes available. Expanding window regressions do not contain look-ahead bias as they only use information that was available for the investor at time t. Out z-scores are also adjusted for look-ahead bias as they are calculated on an expanding window as well. The downside of all this is that we have lost about 10 years of data (warm-up expanding z-score is 60 months and then we use a warm-up of 60 months again to warm-up our predictive regression)

- Since we have estimated coefficients for each month - and for each variable - we can now use the regression results and forecast the log excess return of next month (forecast for t+1 at month t) per predictor. Our forecast is:

$$ \hat r_{t+1} = \hat C_{t} + \hat \beta_{t}x_{t} $$


- Calculate the Campbell and Thompson out-of-sample R-squared statistic for each predictor and for the C-mean:
Project unfinished here!



- built 3 models with signals and weights:
    - binary_model: binary signals: if z-score < 0: invest 100% in treasuries, else 100% S&P500.
    - dynamic_model: weights equal to the percentile of the z-score (2y warm-up)
    - dynamic_step_model: weights equal to the percentile of the z-score but then adjusted to only transact when 10% weight barriers are crossed.
- backtest the 3 models against a static 50/50 portfolio (50% spx and 50% treasuries)

#### 1.3 MOTIVATION
- I publish my work and learning process because of the protégé effect.
- Teaching and/or explaining a process to others is the best way to learn for myself.
#### 1.4 INTENDED USE
- This notebook is for illustrative and education purposes only. Feel free to use this code for these purposes.
- The logic and/or python code might contain errors.All mistakes remain the author's fault. If you find any, let me know so I can rectify.
#### 1.5 LIMITATIONS & FUTURE CHALLENGES
- Lasso, Ridge regression
- other variables
#### 1.6 CREDITS
- This code uses my own package called my_risk_kit.py for certain functions. 
- The idea for the trading model - to use the mean of univariate forecasts instead of multivariate forecasts - came from the interesting paper below:

Rapach, David and Zhou, Guofu, Asset Pricing: Time-Series Predictability (March 24, 2022). Oxford Research Encyclopedia of Economics and Finance, Available at SSRN: https://ssrn.com/abstract=3941499 or http://dx.doi.org/10.2139/ssrn.3941499
