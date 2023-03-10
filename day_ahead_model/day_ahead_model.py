import datetime as dt
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import glm

import yfinance as yf


# %% function that will be used
def confusion_matrix(actual, predictions):
    """ Creates confusion matrix of actual directions versus predicted directions.

    Args:
        actual(Series): The actual directions in a Series.
        predictions(Series): The predicted directions in a Series.
    Returns:
        DataFrame: The confusion matrix as a DataFrame
    """
    actual_trans = ['up' if proba > 0 else 'down' for proba in actual]
    predictions_trans = ['up' if proba > 0.5 else 'down' for proba in predictions]

    conf_matrix = pd.crosstab(pd.Series(actual_trans), pd.Series(predictions_trans),
                              rownames=['actual'], colnames=['Predicted'])
    return conf_matrix


# %% import spx data
weeks_back = 1040
today = dt.datetime.today()
start_date = today - dt.timedelta(weeks=weeks_back)
end_date = dt.datetime.today() - dt.timedelta(days=1)
index_ticker = ['^GSPC']
spx = yf.download(tickers=index_ticker, start=start_date, end=end_date)['Adj Close']
spx_data = spx.pct_change()
spx_data.name = 'spx_today'

spx_data = spx_data.to_frame()

# %% create spx lag variables in data

for i in range(1, 4):
    spx_data['spx_lag' + str(i)] = spx_data.spx_today.shift(i)

# %% import vix data
vix_ticker = ['^VIX']
vix = yf.download(tickers=vix_ticker, start=start_date, end=end_date)['Adj Close']
vix_data = vix.diff()
vix_data.name = 'vix_today'

vix_data = vix_data.to_frame()

# %% create vix laf variables in vix_data

for i in range(1, 4):
    vix_data['vix_lag' + str(i)] = vix_data.vix_today.shift(i)

# %% merge vix and spx data
data = pd.merge_ordered(spx_data, vix_data, on='Date', how='left')
data.set_index('Date', inplace=True)

data = data.dropna()

# %% up/down column

data['direction'] = np.where(data.spx_today > 0, 1, 0)

# %% save to a csv file
data.to_csv('logistic_data.csv')

# %% define train data and run logistic regression
train_data = data.loc[:'2017-12-31']
train_model = glm(formula="direction ~ spx_lag1 + spx_lag2 + spx_lag3 + vix_lag1 + vix_lag2 + vix_lag3",
                  data=train_data, family=sm.families.Binomial()).fit()
train_model.summary()

train_predictions = train_model.predict(train_data)

# %%  confusion_matrix for training data (in_sample) with metrics
train_confusion_matrix = confusion_matrix(train_data.direction, train_predictions)
train_accuracy = ((train_confusion_matrix.values[0, 0] + train_confusion_matrix.values[1, 1]) /
                  train_confusion_matrix.values.sum())

print(f'training_accuracy is {train_accuracy * 100:.2f}% (Correct predictions: up and down)')
train_tpr = (train_confusion_matrix.values[1, 1] / (train_confusion_matrix.values[1, 0] +
                                                    train_confusion_matrix.values[1, 1]))
print(f'training_tpr is {train_tpr * 100:.2f}% (Up-days predicted versus all up-days)')
train_tnr = (train_confusion_matrix.values[0, 0] / (train_confusion_matrix.values[0, 0] +
                                                    train_confusion_matrix.values[0, 1]))
print(f'training_tnr is {train_tnr * 100:.2f}% (Down-days predicted versus all down-days)')

# %% predict out of sample on test data and run the confusion matrix
test_data = data.loc['2018-01-02':]
test_predictions = train_model.predict(test_data)

test_confusion_matrix = confusion_matrix(test_data.direction, test_predictions)

test_accuracy = ((test_confusion_matrix.values[0, 0] + test_confusion_matrix.values[1, 1]) /
                 test_confusion_matrix.values.sum())
print(f'test_accuracy is {test_accuracy * 100:.2f}% (Correct predictions: up and down)')
test_tpr = (test_confusion_matrix.values[1, 1] / (test_confusion_matrix.values[1, 0] +
                                                  test_confusion_matrix.values[1, 1]))
print(f'test_tpr is {test_tpr * 100:.2f}% (Up-days predicted versus all up-days)')
test_tnr = (test_confusion_matrix.values[0, 0] / (test_confusion_matrix.values[0, 0] +
                                                  test_confusion_matrix.values[0, 1]))
print(f'test_tnr is {test_tnr * 100:.2f}% (Down-days predicted versus all down-days)')

# %% create simple long-short trading strategy on the out of sample
oos_predictions_binary = test_predictions.apply(lambda x: 1 if x > 0.5 else 0)

transaction_cost = 0.001
trade_condition = (oos_predictions_binary != oos_predictions_binary.shift(1))
transaction_costs = pd.Series([transaction_cost if trade is True else 0 for trade
                               in trade_condition], index=trade_condition.index)


oos_benchmark_cum_returns = (1 + test_data.spx_today).cumprod()
oos_benchmark_cum_returns.name = 'S&P500'
oos_strategy_cum_gross_returns = (1 + oos_predictions_binary*test_data.spx_today).cumprod()
oos_strategy_cum_gross_returns.name = 'long-short_strategy_gross'
oos_strategy_cum_net_returns = (1 + (oos_predictions_binary*test_data.spx_today-transaction_costs)).cumprod()
oos_strategy_cum_net_returns.name = 'long_short_strategy_net_'

oos_cum_returns = pd.concat([oos_benchmark_cum_returns, oos_strategy_cum_gross_returns,
                             oos_strategy_cum_net_returns], axis=1)
# %% plot
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('''Out of sample strategy: 2018 - now.
logistic regression (training sample: 2003 - 2017):  S&P500 daily direction on lagged daily S&P500 returns
and vix changes of order 2.
strategy: long IF predicted return is positive ELSE short''', fontsize=10)

oos_cum_returns.plot(ax=ax1)
ax2.plot(oos_strategy_cum_net_returns / oos_benchmark_cum_returns)

sns.despine()

plt.show()
