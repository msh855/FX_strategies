# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
import scipy.stats as stats
import openpyxl


# %% user_defined functions used in this script

def expanding_z_score(seq, warmup=60):
    """ Returns a Series or  DataFrame of z-scores calculated on an expanding window with a warmup period.

    Args:
        seq(Series or DataFrame): a sequence of values for which a z-score will be calculated on the expanding window.
        warmup(int): The warm-up period is the first period that is used to calculate the first z-score. After that
        the z-score is calculated on an expanding window of the sequence: from the very first start value until the
         next step.
    Returns:
        Series or DataFrame: z-scores calculated on an expanding window of values.
    """
    seq = seq.dropna()
    average_expanding = seq.expanding(min_periods=warmup).mean()  # expanding mean
    std_expanding = seq.expanding(min_periods=warmup).std()  # expanding stdev
    z_expanding = (seq - average_expanding) / std_expanding  # the expanding z-score
    return z_expanding


# %% user defined function for time lagged cross correlations

def cross_correlation(target, feature, lag=0):
    """ Returns tuple(s) of Pearson cross-correlation with their p-values of 2 series of
     equal length for a given lag in the feature.

    Args:
        target (Series):Time-series of equal length as feature. Time-series should be stationary (differenced)
        feature (Series): Time-series of equal length. Time-series should be stationary (differenced)
        lag(int): Lag for the feature time-series
    Returns:
        List: List of tuple(s) holding the Pearson correlation statistic and the p-value


    """
    feature_shifted = feature.shift(lag).dropna()  # create the lag and drop resulting nan
    target_series = target.iloc[lag:]  # make target_series of equal length again with lagged feature

    return stats.pearsonr(target_series, feature_shifted)


# %% import the xlsx sheets

data = pd.read_excel('fx_beer_data.xlsx', engine='openpyxl',
                     sheet_name=['fx', 'tot', 'gfc', 'yield', 'cpi', 'prod'])

# %% global variables
start_date_panel = '1996-04-30'
end_date_panel = '2023-02-28'

# %% construct df for fx prices: transform fx prices (make aligned time series, resample eom, naming,
# base_fx/quote_fx, log of pairs)

fx_data = data['fx'].set_index('date')
fx_data = fx_data.resample('M').last()

# transform all pairs to format base_fx/quote_fx with quote_fx always being usd as reference currency
fx_data['cadusd'] = 1 / fx_data.usdcad
fx_data['jpyusd'] = 1 / fx_data.usdjpy
fx_data['sekusd'] = 1 / fx_data.usdsek
fx_data['nokusd'] = 1 / fx_data.usdnok
fx_data['chfusd'] = 1 / fx_data.usdchf
fx_data['plnusd'] = 1 / fx_data.usdpln
fx_data['hufusd'] = 1 / fx_data.usdhuf
fx_data['czkusd'] = 1 / fx_data.usdczk

# list of tickers needed in final df

g12_tickers_conv = ['eurusd', 'usdcad', 'usdjpy', 'gbpusd', 'usdsek', 'usdnok', 'usdchf', 'audusd',
                    'nzdusd', 'usdpln', 'usdhuf', 'usdczk']
g12_tickers = ['eurusd', 'cadusd', 'jpyusd', 'gbpusd', 'sekusd', 'nokusd', 'chfusd', 'audusd', 'nzdusd',
               'plnusd', 'hufusd', 'czkusd']
g9_tickers = ['eurusd', 'cadusd', 'jpyusd', 'gbpusd', 'sekusd', 'nokusd', 'chfusd', 'audusd', 'nzdusd']
cee3_tickers = ['plnusd', 'hufusd', 'czkusd']

# final df for fx prices in log format and correct time_series format
fx = fx_data[g12_tickers]
fx_log = np.log(fx)
fx_log = fx_log.loc[start_date_panel:]

print(f'fx_log shape of frame is {fx_log.shape}, \n'
      f'fx_log starts at {fx_log.index.date[0]}, \n'
      f'fx_log ends at {fx_log.index.date[-1]}')

# %% plot of conventional quoted fx_pairs

fig, ax = plt.subplots(nrows=12, figsize=(7, 9), sharex=True)
fig.suptitle('FX evolution since 1996: market convention quotes', fontsize=10)
plt.rcParams['font.size'] = 7
for index, currency in enumerate(g12_tickers_conv):
    ax[index].plot(fx_data[currency], label=currency)
    ax[index].legend(loc='upper left', fontsize=7)

sns.despine()
plt.savefig('fx_chart_convention.png', dpi=300)
plt.show()

# %% plot of fx_pairs with usd as the quoted currency

fig, ax = plt.subplots(nrows=12, figsize=(7, 9), sharex=True)
fig.suptitle('FX evolution since 1996: usd as quoted currency', fontsize=10)
plt.rcParams['font.size'] = 7
for index, currency in enumerate(g12_tickers):
    ax[index].plot(fx_data[currency], label=currency)
    ax[index].legend(loc='upper left', fontsize=7)
sns.despine()
plt.savefig('fx_chart.png', dpi=300)
plt.show()

# %% construct df for relative terms of trade (tot)

tot_data = data['tot'].set_index('date')
tot_data = tot_data.resample('M').last()

tot_index = tot_data.copy()
tot_index = tot_index + 100
tot_ratio = tot_index.iloc[:, 1:].div(tot_index.usd_tot, axis=0)  # tot relative to USA
tot_log_ratio = np.log(tot_ratio).shift(1).dropna()  # take log and shift a month for point in time issues
tot_log_ratio = tot_log_ratio.loc[start_date_panel:]

print(f'tot_log_ratio shape of frame is {tot_log_ratio.shape}, \n'
      f'tot_log_ratio starts at {tot_log_ratio.index.date[0]}, \n'
      f'tot_log_ratio ends at {tot_log_ratio.index.date[-1]}')

# %% plot the relative tot of each currency for data exploration

fig, ax = plt.subplots(nrows=12, figsize=(7, 9), sharex=True)
fig.suptitle('Relative terms of trade evolution since 1996', fontsize=10)
plt.rcParams['font.size'] = 7
for index, tot in enumerate(tot_ratio.columns):
    ax[index].plot(tot_ratio[tot], label=tot)
    ax[index].legend(loc='upper left', fontsize=7)
sns.despine()
plt.savefig('relative_tot_chart.png', dpi=300)
plt.show()

# %% plot scatter plots of fx_log versus tot_log_ratio

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(7, 9))
fig.suptitle('Scatter plots of log(fx) versus log(relative terms of trade)', fontsize=10)
plt.rcParams['font.size'] = 7
for index, tot in enumerate(tot_log_ratio.columns):
    if index <= 5:
        sns.scatterplot(ax=ax[index, 0], x=tot_log_ratio[tot], y=fx_log.iloc[:, index], label=tot)
        ax[index, 0].legend(loc='lower right', fontsize=7)
    else:
        sns.scatterplot(ax=ax[(index - 6), 1], x=tot_log_ratio[tot], y=fx_log.iloc[:, index], label=tot)
        ax[(index - 6), 1].legend(loc='lower right', fontsize=7)
sns.despine()
plt.savefig('scatter_fx_tot.png', dpi=300)
plt.show()

# %% construct df for relative gfc

gfc_data = data['gfc'].set_index('date')
gfc_data.index = pd.to_datetime(gfc_data.index, format='%Y')
gfc_data = gfc_data.resample('A').last()  # resample to end of year
gfc_data_monthly = gfc_data.resample('M').last().ffill()  # resample eom & ffill missing
gfc_new_date_range = pd.date_range(start='1996-01-31', end=end_date_panel, freq='M')
gfc_reindexed = gfc_data_monthly.reindex(gfc_new_date_range, method='ffill')
gfc_ratio = gfc_reindexed.iloc[:, 1:].div(gfc_reindexed.usd_gfc, axis=0)  # gfc relative to USA
gfc_log_ratio = np.log(gfc_ratio)  # point in time not necessary, already year lag tsss
gfc_log_ratio = gfc_log_ratio.loc[start_date_panel:]

print(f'gfc_log_ratio shape of frame is {gfc_log_ratio.shape}, \n'
      f'gfc_log_ratio starts at {gfc_log_ratio.index.date[0]}, \n'
      f'gfc_log_ratio ends at {gfc_log_ratio.index.date[-1]}')

# %% plot the relative gfc of each currency for data exploration

fig, ax = plt.subplots(nrows=12, figsize=(7, 9), sharex=True)
fig.suptitle('Relative Gross Fixed Capital % GDP evolution since 1996', fontsize=10)
plt.rcParams['font.size'] = 7
for index, gfc in enumerate(gfc_ratio.columns):
    ax[index].plot(gfc_ratio[gfc], label=gfc)
    ax[index].legend(loc='upper left', fontsize=7)
sns.despine()
plt.savefig('relative_gfc_chart.png', dpi=300)
plt.show()

# %% plot scatter plots of fx_log versus gfc_log_ratio

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(7, 9))
fig.suptitle('Scatter plots of log(fx) versus log(relative Gross Fixed Capital as % GDP)', fontsize=10)
plt.rcParams['font.size'] = 7
for index, gfc in enumerate(gfc_log_ratio.columns):
    if index <= 5:
        sns.scatterplot(ax=ax[index, 0], x=gfc_log_ratio[gfc], y=fx_log.iloc[:, index], label=gfc)
        ax[index, 0].legend(loc='lower right', fontsize=7)
    else:
        sns.scatterplot(ax=ax[(index - 6), 1], x=gfc_log_ratio[gfc], y=fx_log.iloc[:, index], label=gfc)
        ax[(index - 6), 1].legend(loc='lower right', fontsize=7)
sns.despine()
plt.savefig('scatter_fx_gfc.png', dpi=300)
plt.show()

# %% construct df for yield for g9 and cee3 (different starting dates which in the end will create unbalanced panel)

g9_yield_data = data['yield'].iloc[:, :11].set_index('date')
g9_yield_data_monthly = g9_yield_data.resample('M').last()
g9_yield_diff = g9_yield_data_monthly.iloc[:, 1:].sub(g9_yield_data_monthly.usd_yield, axis=0)  # yield diff with  USA
g9_yield_diff = g9_yield_diff.loc[start_date_panel:]

print(f'g9_yield_diff shape of frame is {g9_yield_diff.shape}, \n'
      f'g9_yield_diff starts at {g9_yield_diff.index.date[0]}, \n'
      f'g9_yield_diff ends at {g9_yield_diff.index.date[-1]}')

cee3_yield_data = data['yield'].iloc[:, 11:15].set_index('date.1')
cee3_yield_data.index.name = 'date'
cee3_yield_data_monthly = cee3_yield_data.resample('M').last()
cee3_yield_diff = cee3_yield_data_monthly.sub(g9_yield_data_monthly['2001':].usd_yield, axis=0)  # yield diff with  USA
cee3_yield_diff = cee3_yield_diff.loc['2001-01-31':]

# point in time not necessary here

print(f'cee3_yield_diff shape of frame is {cee3_yield_diff.shape}, \n'
      f'cee3_yield_diff starts at {cee3_yield_diff.index.date[0]}, \n'
      f'cee3_yield_diff ends at {cee3_yield_diff.index.date[-1]}')

# %% plot the yield_diff of each currency for data exploration

fig, ax = plt.subplots(nrows=12, figsize=(6, 9), sharex=True)
fig.suptitle('Yield Differential with USA since 1996', fontsize=10)
plt.rcParams['font.size'] = 7
for index, yield_diff in enumerate(g9_yield_diff.columns):
    ax[index].plot(g9_yield_diff[yield_diff], label=yield_diff)
    ax[index].legend(loc='upper left', fontsize=7)
for index, yield_diff in enumerate(cee3_yield_diff.columns):
    ax[index + 9].plot(cee3_yield_diff[yield_diff], label=yield_diff)
    ax[index + 9].legend(loc='upper left', fontsize=7)
sns.despine()
plt.savefig('yield_diff_chart.png', dpi=300)
plt.show()

# %% plot scatter plots of fx_log versus yield_diff

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(7, 9))
fig.suptitle('Scatter plots of log(fx) versus 10y yield differentials', fontsize=10)
plt.rcParams['font.size'] = 7
for index, yield_diff in enumerate(g9_yield_diff.columns):
    if index <= 5:
        sns.scatterplot(ax=ax[index, 0], x=g9_yield_diff[yield_diff], y=fx_log.iloc[:, index], label=yield_diff)
        ax[index, 0].legend(loc='lower right', fontsize=7)
    else:
        sns.scatterplot(ax=ax[(index - 6), 1], x=g9_yield_diff[yield_diff], y=fx_log.iloc[:, index], label=yield_diff)
        ax[(index - 6), 1].legend(loc='lower right', fontsize=7)
for index, yield_diff in enumerate(cee3_yield_diff.columns):
    if index <= 5:
        sns.scatterplot(ax=ax[index + 3, 1], x=cee3_yield_diff[yield_diff], y=fx_log.iloc[:, index + 9],
                        label=yield_diff)
        ax[index + 3, 1].legend(loc='lower right', fontsize=7)

sns.despine()
plt.savefig('scatter_fx_yielddiff.png', dpi=300)
plt.show()

# %% construct df for log cpi ratio

cpi_data = data['cpi'].iloc[:, :12].set_index('date')
cpi_data = cpi_data.resample('M').last()
cpi_new_date_range = pd.date_range(start='1996-01-31', end=end_date_panel, freq='M')
cpi_reindexed = cpi_data.reindex(cpi_new_date_range)
cpi_reindexed = cpi_reindexed.ffill()
cpi_ratio = cpi_reindexed.iloc[:, 1:].div(cpi_reindexed.usd_cpi, axis=0)  # relative cpi
cpi_log_ratio = np.log(cpi_ratio)  # take log, wait to shift for point in time, add nzd & aud first

cpi_data_audnzd = data['cpi'].iloc[:, 12:].set_index('date.1')
cpi_data_audnzd = cpi_data_audnzd.loc[:'2022-12-30']
cpi_data_audnzd_monthly = cpi_data_audnzd.resample('M').last().ffill()
cpi_data_audnzd_monthly_reindexed = cpi_data_audnzd_monthly.reindex(cpi_new_date_range)
cpi_data_audnzd_monthly_reindexed = cpi_data_audnzd_monthly_reindexed.ffill()
cpi_data_audnzd_ratio = cpi_data_audnzd_monthly_reindexed.div(cpi_reindexed.usd_cpi, axis=0)
cpi_audnzd_log_ratio = np.log(cpi_data_audnzd_ratio)

cpi_log_ratio = cpi_log_ratio.join(cpi_audnzd_log_ratio)
cpi_log_ratio = cpi_log_ratio.shift(1).dropna()  # make point in time and lag a month
cpi_log_ratio = cpi_log_ratio[['eur_cpi', 'cad_cpi', 'jpy_cpi', 'gbp_cpi', 'sek_cpi',
                               'nok_cpi', 'chf_cpi', 'aud_cpi', 'nzd_cpi', 'pln_cpi',
                               'huf_cpi', 'czk_cpi']]  # reorder columns just for my sake of mind
cpi_log_ratio = cpi_log_ratio.loc[start_date_panel:]

print(f'cpi_log_ratio shape of frame is {cpi_log_ratio.shape}, \n'
      f'cpi_log_ratio starts at {cpi_log_ratio.index.date[0]}, \n'
      f'cpi_log_ratio ends at {cpi_log_ratio.index.date[-1]}')

# %% plot the relative cpi of each currency for data visualisation

fig, ax = plt.subplots(nrows=12, figsize=(7, 9), sharex=True)
fig.suptitle('Relative CPI index versus USA since 1996', fontsize=10)
plt.rcParams['font.size'] = 7
for index, cpi in enumerate(cpi_log_ratio.columns):
    ax[index].plot(cpi_log_ratio[cpi], label=cpi)
    ax[index].legend(loc='upper left', fontsize=7)
sns.despine()
plt.savefig('relative_cpi_chart.png', dpi=300)
plt.show()

# %% plot scatter plots of fx_log versus cpi_log_ratio

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(7, 9))
fig.suptitle('Scatter plots of log(fx) versus log(relative CPI-index)', fontsize=10)
plt.rcParams['font.size'] = 7
for index, cpi in enumerate(cpi_log_ratio.columns):
    if index <= 5:
        sns.scatterplot(ax=ax[index, 0], x=cpi_log_ratio[cpi], y=fx_log.iloc[:, index], label=cpi)
        ax[index, 0].legend(loc='lower right', fontsize=7)
    else:
        sns.scatterplot(ax=ax[(index - 6), 1], x=cpi_log_ratio[cpi], y=fx_log.iloc[:, index], label=cpi)
        ax[(index - 6), 1].legend(loc='lower right', fontsize=7)
sns.despine()
plt.savefig('scatter_fx_cpi.png', dpi=300)
plt.show()

# %% construct df for log relative productivity index

prod_data = data['prod'].set_index('date')
prod_data = prod_data.resample('M').last().ffill()

prod_new_date_range = pd.date_range(start='1996-01-31', end=end_date_panel, freq='M')
prod_reindexed = prod_data.reindex(prod_new_date_range)
prod_reindexed = prod_reindexed.ffill()
prod_ratio = prod_reindexed.iloc[:, 1:].div(prod_reindexed.usd_prod, axis=0)  # relative to us
prod_log_ratio = np.log(prod_ratio).shift(3).dropna()  # quarter lag

print(f'prod_log_ratio shape of frame is {prod_log_ratio.shape}, \n'
      f'prod_log_ratio starts at {prod_log_ratio.index.date[0]}, \n'
      f'prod_log_ratio ends at {prod_log_ratio.index.date[-1]}')

# %% plot the relative productivity index of each currency for data visualisation

fig, ax = plt.subplots(nrows=12, figsize=(7, 9), sharex=True)
fig.suptitle('Relative labour productivity index versus USA since 1996', fontsize=10)
plt.rcParams['font.size'] = 7
for index, prod in enumerate(prod_ratio.columns):
    ax[index].plot(prod_ratio[prod], label=prod)
    ax[index].legend(loc='upper left', fontsize=7)
sns.despine()
plt.savefig('relative_prod_chart.png', dpi=300)
plt.show()

# %% plot scatter plots of fx_log versus prod_log_ratio

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(7, 9))
fig.suptitle('Scatter plots of log(fx) versus log(relative productivity)', fontsize=10)
plt.rcParams['font.size'] = 7
for index, prod in enumerate(prod_log_ratio.columns):
    if index <= 5:
        sns.scatterplot(ax=ax[index, 0], x=prod_log_ratio[prod], y=fx_log.iloc[:, index], label=prod)
        ax[index, 0].legend(loc='lower right', fontsize=7)
    else:
        sns.scatterplot(ax=ax[(index - 6), 1], x=prod_log_ratio[prod], y=fx_log.iloc[:, index], label=prod)
        ax[(index - 6), 1].legend(loc='lower right', fontsize=7)
sns.despine()
plt.savefig('scatter_fx_prod.png', dpi=300)
plt.show()

# %% create individual g9 & cee3 currency panels with loop, then concat afterwards to complete panel set

g9_panels = []
for index, currency in enumerate(g9_tickers):
    panel = fx_log[[g9_tickers[index]]].copy()
    panel.rename(columns={currency: 'fx_log'}, inplace=True)
    panel['currency'] = currency
    panel = panel[['currency', 'fx_log']]
    panel['tot_log_ratio'] = tot_log_ratio.iloc[:, index]
    panel['gfc_log_ratio'] = gfc_log_ratio.iloc[:, index]
    panel['yield_diff'] = g9_yield_diff.iloc[:, index]
    panel['cpi_log_ratio'] = cpi_log_ratio.iloc[:, index]
    panel['prod_log_ratio'] = prod_log_ratio.iloc[:, index]
    g9_panels.append(panel)

# unpack each panel dataframe and assign to variable
eur_panel, cad_panel, jpy_panel, gbp_panel, sek_panel, nok_panel, chf_panel, aud_panel, nzd_panel = g9_panels

cee3_panels = []
for index, currency in enumerate(cee3_tickers):
    panel = fx_log[[cee3_tickers[index]]].copy()
    panel.rename(columns={currency: 'fx_log'}, inplace=True)
    panel['currency'] = currency
    panel = panel[['currency', 'fx_log']]
    panel['tot_log_ratio'] = tot_log_ratio.iloc[:, index + 9]
    panel['gfc_log_ratio'] = gfc_log_ratio.iloc[:, index + 9]
    panel['yield_diff'] = cee3_yield_diff.iloc[:, index]
    panel['cpi_log_ratio'] = cpi_log_ratio.iloc[:, index + 9]
    panel['prod_log_ratio'] = prod_log_ratio.iloc[:, index + 9]
    panel = panel.loc['2001-01-31':]
    cee3_panels.append(panel)

pln_panel, huf_panel, czk_panel = cee3_panels

# construct total panel data
panel_data = pd.concat([eur_panel, cad_panel, jpy_panel, gbp_panel, sek_panel, nok_panel, chf_panel,
                        aud_panel, nzd_panel, pln_panel, huf_panel, czk_panel], axis=0)

print(f'shape of panel_data is {panel_data.shape}')

# %% test for unit root proces (non-stationary) for just one currency as example. All variables

for col in eur_panel.columns[1:]:
    adf = adfuller(eur_panel[col].dropna())
    print(f'{col} has test-statistic of {adf[0]} en p-value of {adf[1]}')

# %% get dummies per currency

dummies = pd.get_dummies(panel_data.currency)
dummies = dummies[['eurusd', 'cadusd', 'jpyusd', 'gbpusd', 'sekusd', 'nokusd', 'chfusd', 'audusd', 'nzdusd',
                   'plnusd', 'hufusd', 'czkusd']]

panel_data_and_dummies = pd.concat([panel_data, dummies], axis=1)
panel_data_and_dummies = panel_data_and_dummies.reset_index()  # make a multiindex df
panel_data_and_dummies.set_index(['currency', 'date'], inplace=True)  # outer index is currency, inner is date
panel_data_and_dummies.sort_index(inplace=True)  # multi-indices work best if they are sorted for slicing later

# %% plot acf of differenced log fx for just one currency:

plot_acf(fx_log.eurusd.diff().dropna(), zero=False, lags=12, alpha=0.05)
plt.show()

# %% plot time-lagged cross pearson correlation between difference response and differenced variables
# time-lagged cross correlation should be on differenced series always!
panel_data_and_dummies.fx_log.diff().dropna()
panel_data_and_dummies.tot_log_ratio.diff().dropna()

tot_log_ratio_cross_correl = [cross_correlation(panel_data_and_dummies.fx_log.diff().dropna(),
                                                panel_data_and_dummies.tot_log_ratio.diff().dropna(),
                                                lag) for lag in range(1, 13)]

gfc_log_ratio_cross_correl = [cross_correlation(panel_data_and_dummies.fx_log.diff().dropna(),
                                                panel_data_and_dummies.gfc_log_ratio.diff().dropna(),
                                                lag) for lag in range(1, 13)]

yield_diff_cross_correl = [cross_correlation(panel_data_and_dummies.fx_log.diff().dropna(),
                                             panel_data_and_dummies.yield_diff.diff().dropna(),
                                             lag) for lag in range(1, 13)]

cpi_log_ratio_cross_correl = [cross_correlation(panel_data_and_dummies.fx_log.diff().dropna(),
                                                panel_data_and_dummies.cpi_log_ratio.diff().dropna(),
                                                lag) for lag in range(1, 13)]

prod_log_ratio_cross_correl = [cross_correlation(panel_data_and_dummies.fx_log.diff().dropna(),
                                                 panel_data_and_dummies.prod_log_ratio.diff().dropna(),
                                                 lag) for lag in range(1, 13)]

cross_correl_df = pd.DataFrame({
    'time_lags': pd.Series([i for i in range(1, 13)]),
    'tot_log_ratio_correl (cor, p)': [np.round((stat[0], stat[1]), 3) for stat in tot_log_ratio_cross_correl],
    'gfc_log_ratio_correl (cor, p)': [np.round((stat[0], stat[1]), 3) for stat in gfc_log_ratio_cross_correl],
    'yield_diff_correl (cor, p)': [np.round((stat[0], stat[1]), 3) for stat in yield_diff_cross_correl],
    'cpi_log_ratio_correl (cor, p)': [np.round((stat[0], stat[1]), 3) for stat in cpi_log_ratio_cross_correl],
    'prod_log_ratio_correl (cor, p)': [np.round((stat[0], stat[1]), 3) for stat in prod_log_ratio_cross_correl]
})

# %% perform the in_sample fixed effect panel regression through lsdv (least-squares dummy variable)

formula = 'fx_log ~ tot_log_ratio + gfc_log_ratio + yield_diff + cpi_log_ratio + prod_log_ratio +' \
          'cadusd + jpyusd + gbpusd +sekusd + nokusd + chfusd + audusd + nzdusd + plnusd + hufusd + czkusd'

is_model = ols(formula=formula, data=panel_data_and_dummies).fit()
print(is_model.summary())

# %% test for cointegration using Engle-Granger: test for stationarity on the estimated residuals
# of the regression that estimates the long run relationship between fx and variables
# the residuals are nothing more than the deviations from fair value

residual_stationary_test = adfuller(is_model.resid)
print(f'The adf-test on the residuals of our panel regression has a test-statistic '
      f'of {residual_stationary_test[0]} and a p-value of {residual_stationary_test[1]}')

# %% in_sample predictions: fitted values

# multi-index series with fair values predicted in_sample (is)
is_predictions = is_model.predict(panel_data_and_dummies)

# multi-index df
is_predictions_df = panel_data_and_dummies[['fx_log']].copy()
is_predictions_df['fx_log_fair_is'] = is_predictions

# %% vector error correction model: only cross correl was lag 6 for tot
# p_values for monthly are not significant, so we calculate quarterly like UniCredit

response_vecm = panel_data_and_dummies.fx_log.diff(3).dropna()  # response is first difference at time t
response_vecm.name = 'diff_fx_log'

predictors_vecm = panel_data_and_dummies[['tot_log_ratio']]
predictors_vecm = sm.add_constant(predictors_vecm)  # add a constant

# 6-lag change in tot was relevant in cross-correlation matrix: take diff, lag it 6 periods
predictors_vecm.loc[:, 'tot_log_ratio'] = predictors_vecm['tot_log_ratio'].diff().shift(6)

# add the residuals from the panel regression and lag them 1 period
predictors_vecm['lagged residuals'] = is_model.resid.shift(3)
predictors_vecm = predictors_vecm.dropna()

# align response again
response_vecm = response_vecm.tail(-4)


# %% the  vecm model
vecm_model = sm.OLS(response_vecm, predictors_vecm).fit()
print(vecm_model.summary())

# %% out_of_sample prediction warmup

WARMUP = 59  # 60 months warmup for expanding regression

# %% out_of_sample expanding regressions and fair_value predictions for g9 currencies

# Construct panel data for g9 only that will be used for expanding regression & prediction
panel_data_and_dummies_g9 = panel_data_and_dummies.loc[(g9_tickers, slice(None)), :].copy()
panel_data_and_dummies_g9 = panel_data_and_dummies_g9.sort_index()

# construct df for out of sample predictions: holds actual and will hold fair values based on expanding window
oos_predictions_g9_df = panel_data_and_dummies.loc[(g9_tickers, slice(None)), ['fx_log']].copy()
oos_predictions_g9_df = oos_predictions_g9_df.sort_index()

# every period perform a regression on expanding window (warmup 60m), predict for that month, append prediction
first_date = panel_data_and_dummies_g9.index.get_level_values(1)[0]

for i in range(WARMUP, 323):
    rolling_end_date = panel_data_and_dummies_g9.index.get_level_values(1)[i]
    expanding_panel = panel_data_and_dummies_g9.loc[(slice(None), slice(first_date, rolling_end_date)), :]

    oos_model = ols(formula=formula, data=expanding_panel).fit()
    oos_prediction_for_one_date = oos_model.predict(panel_data_and_dummies_g9.loc[(slice(None), rolling_end_date), :])
    oos_prediction_for_one_date = oos_prediction_for_one_date.to_frame(name=rolling_end_date)
    oos_predictions_g9_df.loc[(slice(None), rolling_end_date), 'fx_log_fair_oos'] = oos_prediction_for_one_date.values

# %% out_of_sample expanding regressions and fair_value predictions for cee3 currencies

# Construct panel data for cee3 only that will be used for expanding regression & prediction
panel_data_and_dummies_cee3 = panel_data_and_dummies.loc[(cee3_tickers, slice(None)), :].copy()
panel_data_and_dummies_cee3 = panel_data_and_dummies_cee3.sort_index()

# construct df for out of sample predictions: holds actual and will hold fair values based on expanding window
oos_predictions_cee3_df = panel_data_and_dummies.loc[(cee3_tickers, slice(None)), ['fx_log']].copy()
oos_predictions_cee3_df = oos_predictions_cee3_df.sort_index()

# every period perform a regression on expanding window (warmup 60m), predict for that month, append prediction
first_date_oos = panel_data_and_dummies_cee3.index.get_level_values(1)[0]

for index in range(WARMUP, 266):
    rolling_end_date_cee3 = panel_data_and_dummies_cee3.index.get_level_values(1)[index]
    expanding_panel_cee3 = panel_data_and_dummies_cee3.loc[(slice(None), slice(first_date_oos,
                                                                               rolling_end_date_cee3)), :]
    oos_model_cee3 = ols(formula=formula, data=expanding_panel_cee3).fit()
    oos_prediction_for_one_date_cee3 = oos_model_cee3.predict(panel_data_and_dummies_cee3.loc[(slice(None),
                                                                                               rolling_end_date_cee3),
                                                              :])
    oos_prediction_for_one_date_cee3 = oos_prediction_for_one_date_cee3.to_frame(name=rolling_end_date_cee3)
    oos_predictions_cee3_df.loc[
        (slice(None), rolling_end_date_cee3), 'fx_log_fair_oos'] = oos_prediction_for_one_date_cee3.values

# %% construct total oos df: concat g9 and cee3

oos_predictions_df = pd.concat([oos_predictions_g9_df, oos_predictions_cee3_df], axis=0)
oos_predictions_df = oos_predictions_df.sort_index()

# %% fair value for eurusd: chart


end_of_period_values = []
for currency in g12_tickers:
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.tight_layout(pad=5.0)
    fig.suptitle(f'{currency}: actual level and long-term fair values (in-and-out-of-sample, beer model)')
    fx_actual = np.exp(is_predictions_df.loc[(currency, slice(None)), 'fx_log']).droplevel(level=0)
    fx_fair_is = np.exp(is_predictions_df.loc[(currency, slice(None)), 'fx_log_fair_is']).droplevel(level=0)
    fx_fair_oos = np.exp(oos_predictions_df.loc[(currency, slice(None)), 'fx_log_fair_oos']).droplevel(level=0)

    fx_actual.plot(ax=ax[0], label=f'actual {currency}', legend=True)
    fx_fair_is.plot(ax=ax[0], color='orange', label='in-sample, long-term fair value (beer)', legend=True)
    fx_fair_oos.plot(ax=ax[0], color='red', label='out-of-sample (expanding window), long-term fair value (beer)',
                     legend=True)

    deviation_perc_is = fx_actual.div(fx_fair_is).sub(1).mul(100)
    deviation_perc_oos = fx_actual.div(fx_fair_oos).sub(1).mul(100)

    deviation_perc_is.plot(ax=ax[1], label='in-sample % deviation from fair value (beer)', legend=True)
    deviation_perc_oos.plot(ax=ax[1], color='red',
                            label='out-of-sample (expanding window) % deviation from fair value ('
                                  'beer)', legend=True)

    ax[1].set_title(f'{currency}: % deviation versus long-term fair values (in-and-out-of-sample, beer model) (-/+)')
    ax[1].set_ylabel('% deviation (-/+)')
    ax[1].axhline(0, color='red', linestyle='--')

    deviation_z_is = stats.zscore(deviation_perc_is)
    deviation_z_oos = expanding_z_score(deviation_perc_oos)
    deviation_z_is.plot(ax=ax[2], label='in-sample z-score deviation from fair value (beer)', legend=True)
    deviation_z_oos.plot(ax=ax[2], color='red', label='out-of-sample (expanding window) z-score deviation '
                                                      '(expanding window again) from fair value '
                                                      '(beer)', legend=True)
    ax[2].set_title(f'{currency}: z-score deviation versus long-term fair values (in-and-out-of-sample, beer model')
    ax[2].set_ylabel('z-score deviation (-/+)')
    ax[2].axhline(0, color='red', linestyle='--')

    sns.despine()
    plt.show()

    end_of_period_values.append((currency, np.round(deviation_perc_oos[-1], 2), np.round(deviation_z_oos[-1], 2)))


# %% random

end_of_period_df = pd.DataFrame(end_of_period_values, columns=['currency', 'deviation in %', 'deviation in z-score'])
