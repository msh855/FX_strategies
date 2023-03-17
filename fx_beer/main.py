# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
import scipy.stats as stats
import openpyxl

# %% import the xlsx sheets
data = pd.read_excel('fx_beer_data.xlsx', engine='openpyxl',
                     sheet_name=['fx', 'tot', 'gfc', 'yield', 'cpi', 'prod'])

# %% global variables
start_date = '1996-02-29'
end_date = '2023-02-28'

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
fx_log = fx_log.loc['1996-04-30':]

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
tot_log_ratio = tot_log_ratio.loc['1996-04-30':]

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

# %% construct df for relative terms of trade (tot)

gfc_data = data['gfc'].set_index('date')
gfc_data.index = pd.to_datetime(gfc_data.index, format='%Y')
gfc_data = gfc_data.resample('A').last()  # resample to end of year
gfc_data_monthly = gfc_data.resample('M').last().interpolate()  # resample eom & interpolate missing
gfc_new_date_range = pd.date_range(start='1996-01-31', end='2023-02-28', freq='M')
gfc_reindexed = gfc_data_monthly.reindex(gfc_new_date_range, method='ffill')
gfc_ratio = gfc_reindexed.iloc[:, 1:].div(gfc_reindexed.usd_gfc, axis=0)  # gfc relative to USA
gfc_log_ratio = np.log(gfc_ratio)  # point in time not necessary, already year lag tsss
gfc_log_ratio = gfc_log_ratio.loc['1996-04-30':]

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
g9_yield_diff = g9_yield_diff.loc['1996-04-30':]

print(f'g9_yield_diff shape of frame is {g9_yield_diff.shape}, \n'
      f'g9_yield_diff starts at {g9_yield_diff.index.date[0]}, \n'
      f'g9_yield_diff ends at {g9_yield_diff.index.date[-1]}')

cee3_yield_data = data['yield'].iloc[:, 11:15].set_index('date.1')
cee3_yield_data.index.name = 'date'
cee3_yield_data_monthly = cee3_yield_data.resample('M').last()
cee3_yield_diff = cee3_yield_data_monthly.sub(g9_yield_data_monthly['2001':].usd_yield, axis=0)  # yield diff with  USA
cee3_yield_diff = cee3_yield_diff.loc['1996-04-30':]

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
cpi_new_date_range = pd.date_range(start='1996-01-31', end=end_date, freq='M')
cpi_reindexed = cpi_data.reindex(cpi_new_date_range)
cpi_reindexed = cpi_reindexed.ffill()
cpi_ratio = cpi_reindexed.iloc[:, 1:].div(cpi_reindexed.usd_cpi, axis=0)  # relative cpi
cpi_log_ratio = np.log(cpi_ratio)  # take log, wait to shift for point in time, add nzd & aud first

cpi_data_audnzd = data['cpi'].iloc[:, 12:].set_index('date.1')
cpi_data_audnzd = cpi_data_audnzd.loc[:'2022-12-30']
cpi_data_audnzd_monthly = cpi_data_audnzd.resample('M').last().interpolate()
cpi_data_audnzd_monthly_reindexed = cpi_data_audnzd_monthly.reindex(cpi_new_date_range)
cpi_data_audnzd_monthly_reindexed = cpi_data_audnzd_monthly_reindexed.ffill()
cpi_data_audnzd_ratio = cpi_data_audnzd_monthly_reindexed.div(cpi_reindexed.usd_cpi, axis=0)
cpi_audnzd_log_ratio = np.log(cpi_data_audnzd_ratio)

cpi_log_ratio = cpi_log_ratio.join(cpi_audnzd_log_ratio)
cpi_log_ratio = cpi_log_ratio.shift(1).dropna()  # make point in time and lag a month
cpi_log_ratio = cpi_log_ratio[['eur_cpi', 'cad_cpi', 'jpy_cpi', 'gbp_cpi', 'sek_cpi',
                               'nok_cpi', 'chf_cpi', 'aud_cpi', 'nzd_cpi', 'pln_cpi',
                               'huf_cpi', 'czk_cpi']]  # reorder columns just for my sake of mind
cpi_log_ratio = cpi_log_ratio.loc['1996-04-30':]

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
prod_data = prod_data.resample('M').last().interpolate()

prod_new_date_range = pd.date_range(start='1996-01-31', end=end_date, freq='M')
prod_reindexed = prod_data.reindex(prod_new_date_range).ffill()
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

# %% construct individual panel data per currency and then concat on the rows

eur_panel = fx_log[['eurusd']].copy()
eur_panel.rename(columns={'eurusd': 'fx_log'}, inplace=True)
eur_panel['currency'] = 'eurusd'
eur_panel = eur_panel[['currency', 'fx_log']]
eur_panel['tot_log_ratio'] = tot_log_ratio.eur_tot
eur_panel['gfc_log_ratio'] = gfc_log_ratio.eur_gfc
eur_panel['yield_diff'] = g9_yield_diff.eur_yield
eur_panel['cpi_log_ratio'] = cpi_log_ratio.eur_cpi
eur_panel['prod_log_ratio'] = prod_log_ratio.eur_prod

cad_panel: object = fx_log[['cadusd']].copy()
cad_panel.rename(columns={'cadusd': 'fx_log'}, inplace=True)
cad_panel['currency'] = 'cadusd'
cad_panel = cad_panel[['currency', 'fx_log']]
cad_panel['tot_log_ratio'] = tot_log_ratio.cad_tot
cad_panel['gfc_log_ratio'] = gfc_log_ratio.cad_gfc
cad_panel['yield_diff'] = g9_yield_diff.cad_yield
cad_panel['cpi_log_ratio'] = cpi_log_ratio.cad_cpi
cad_panel['prod_log_ratio'] = prod_log_ratio.cad_prod

jpy_panel: object = fx_log[['jpyusd']].copy()
jpy_panel.rename(columns={'jpyusd': 'fx_log'}, inplace=True)
jpy_panel['currency'] = 'jpyusd'
jpy_panel = jpy_panel[['currency', 'fx_log']]
jpy_panel['tot_log_ratio'] = tot_log_ratio.jpy_tot
jpy_panel['gfc_log_ratio'] = gfc_log_ratio.jpy_gfc
jpy_panel['yield_diff'] = g9_yield_diff.jpy_yield
jpy_panel['cpi_log_ratio'] = cpi_log_ratio.jpy_cpi
jpy_panel['prod_log_ratio'] = prod_log_ratio.jpy_prod

gbp_panel: object = fx_log[['gbpusd']].copy()
gbp_panel.rename(columns={'gbpusd': 'fx_log'}, inplace=True)
gbp_panel['currency'] = 'gbpusd'
gbp_panel = gbp_panel[['currency', 'fx_log']]
gbp_panel['tot_log_ratio'] = tot_log_ratio.gbp_tot
gbp_panel['gfc_log_ratio'] = gfc_log_ratio.gbp_gfc
gbp_panel['yield_diff'] = g9_yield_diff.gbp_yield
gbp_panel['cpi_log_ratio'] = cpi_log_ratio.gbp_cpi
gbp_panel['prod_log_ratio'] = prod_log_ratio.gbp_prod

sek_panel: object = fx_log[['sekusd']].copy()
sek_panel.rename(columns={'sekusd': 'fx_log'}, inplace=True)
sek_panel['currency'] = 'sekusd'
sek_panel = sek_panel[['currency', 'fx_log']]
sek_panel['tot_log_ratio'] = tot_log_ratio.sek_tot
sek_panel['gfc_log_ratio'] = gfc_log_ratio.sek_gfc
sek_panel['yield_diff'] = g9_yield_diff.sek_yield
sek_panel['cpi_log_ratio'] = cpi_log_ratio.sek_cpi
sek_panel['prod_log_ratio'] = prod_log_ratio.sek_prod

nok_panel: object = fx_log[['nokusd']].copy()
nok_panel.rename(columns={'nokusd': 'fx_log'}, inplace=True)
nok_panel['currency'] = 'nokusd'
nok_panel = nok_panel[['currency', 'fx_log']]
nok_panel['tot_log_ratio'] = tot_log_ratio.nok_tot
nok_panel['gfc_log_ratio'] = gfc_log_ratio.nok_gfc
nok_panel['yield_diff'] = g9_yield_diff.nok_yield
nok_panel['cpi_log_ratio'] = cpi_log_ratio.nok_cpi
nok_panel['prod_log_ratio'] = prod_log_ratio.nok_prod

chf_panel: object = fx_log[['chfusd']].copy()
chf_panel.rename(columns={'chfusd': 'fx_log'}, inplace=True)
chf_panel['currency'] = 'chfusd'
chf_panel = chf_panel[['currency', 'fx_log']]
chf_panel['tot_log_ratio'] = tot_log_ratio.chf_tot
chf_panel['gfc_log_ratio'] = gfc_log_ratio.chf_gfc
chf_panel['yield_diff'] = g9_yield_diff.chf_yield
chf_panel['cpi_log_ratio'] = cpi_log_ratio.chf_cpi
chf_panel['prod_log_ratio'] = prod_log_ratio.chf_prod

aud_panel: object = fx_log[['audusd']].copy()
aud_panel.rename(columns={'audusd': 'fx_log'}, inplace=True)
aud_panel['currency'] = 'audusd'
aud_panel = aud_panel[['currency', 'fx_log']]
aud_panel['tot_log_ratio'] = tot_log_ratio.aud_tot
aud_panel['gfc_log_ratio'] = gfc_log_ratio.aud_gfc
aud_panel['yield_diff'] = g9_yield_diff.aud_yield
aud_panel['cpi_log_ratio'] = cpi_log_ratio.aud_cpi
aud_panel['prod_log_ratio'] = prod_log_ratio.aud_prod

nzd_panel: object = fx_log[['nzdusd']].copy()
nzd_panel.rename(columns={'nzdusd': 'fx_log'}, inplace=True)
nzd_panel['currency'] = 'nzdusd'
nzd_panel = nzd_panel[['currency', 'fx_log']]
nzd_panel['tot_log_ratio'] = tot_log_ratio.nzd_tot
nzd_panel['gfc_log_ratio'] = gfc_log_ratio.nzd_gfc
nzd_panel['yield_diff'] = g9_yield_diff.nzd_yield
nzd_panel['cpi_log_ratio'] = cpi_log_ratio.nzd_cpi
nzd_panel['prod_log_ratio'] = prod_log_ratio.nzd_prod

pln_panel: object = fx_log[['plnusd']].copy()
pln_panel.rename(columns={'plnusd': 'fx_log'}, inplace=True)
pln_panel['currency'] = 'plnusd'
pln_panel = pln_panel[['currency', 'fx_log']]
pln_panel['tot_log_ratio'] = tot_log_ratio.pln_tot
pln_panel['gfc_log_ratio'] = gfc_log_ratio.pln_gfc
pln_panel['yield_diff'] = cee3_yield_diff.pln_yield
pln_panel['cpi_log_ratio'] = cpi_log_ratio.pln_cpi
pln_panel['prod_log_ratio'] = prod_log_ratio.pln_prod
pln_panel = pln_panel.loc['2001-01-31':]

huf_panel: object = fx_log[['hufusd']].copy()
huf_panel.rename(columns={'hufusd': 'fx_log'}, inplace=True)
huf_panel['currency'] = 'hufusd'
huf_panel = huf_panel[['currency', 'fx_log']]
huf_panel['tot_log_ratio'] = tot_log_ratio.huf_tot
huf_panel['gfc_log_ratio'] = gfc_log_ratio.huf_gfc
huf_panel['yield_diff'] = cee3_yield_diff.huf_yield
huf_panel['cpi_log_ratio'] = cpi_log_ratio.huf_cpi
huf_panel['prod_log_ratio'] = prod_log_ratio.huf_prod
huf_panel = huf_panel.loc['2001-01-31':]

czk_panel: object = fx_log[['czkusd']].copy()
czk_panel.rename(columns={'czkusd': 'fx_log'}, inplace=True)
czk_panel['currency'] = 'czkusd'
czk_panel = czk_panel[['currency', 'fx_log']]
czk_panel['tot_log_ratio'] = tot_log_ratio.czk_tot
czk_panel['gfc_log_ratio'] = gfc_log_ratio.czk_gfc
czk_panel['yield_diff'] = cee3_yield_diff.czk_yield
czk_panel['cpi_log_ratio'] = cpi_log_ratio.czk_cpi
czk_panel['prod_log_ratio'] = prod_log_ratio.czk_prod
czk_panel = czk_panel.loc['2001-01-31':]

# construct total panel data
panel_data = pd.concat([eur_panel, cad_panel, jpy_panel, gbp_panel, sek_panel, nok_panel, chf_panel,
                        aud_panel, nzd_panel, pln_panel, huf_panel, czk_panel], axis=0)

print(f'shape of panel_data is {panel_data.shape}')

# %% get dummies per currency

dummies = pd.get_dummies(panel_data.currency)
dummies = dummies[['eurusd', 'cadusd', 'jpyusd', 'gbpusd', 'sekusd', 'nokusd', 'chfusd', 'audusd', 'nzdusd',
                   'plnusd', 'hufusd', 'czkusd']]

panel_data_and_dummies = pd.concat([panel_data, dummies], axis=1)

# %% perform the in_sample fixed effect panel regression through lsdv (least-squares dummy variable)

formula = 'fx_log ~ tot_log_ratio + gfc_log_ratio + yield_diff + cpi_log_ratio + prod_log_ratio +' \
          'cadusd + jpyusd + gbpusd +sekusd + nokusd + chfusd + audusd + nzdusd + plnusd + hufusd + czkusd'

in_sample_model = ols(formula=formula, data=panel_data_and_dummies).fit()
print(in_sample_model.summary())

# %% fair value for currencies: latest levels and chart

eurusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'eurusd']))
eurusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'eurusd'][['fx_log']])
eurusd_fair['predicted_eurusd'] = eurusd_pred
eurusd_fair.columns = ['actual_eurusd', 'fair_eurusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('EURUSD: actual level and long-term fair value (beer)')
eurusd_fair.actual_eurusd.plot(ax=ax[0], label='actual eurusd', legend=True)
eurusd_fair.fair_eurusd.plot(ax=ax[0], color='orange', label='fair value (beer model', legend=True)

deviation_perc = eurusd_fair.actual_eurusd.div(eurusd_fair.fair_eurusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('EURUSD: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation)
deviation_z.plot(ax=ax[2])
ax[2].set_title('EURUSD: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for eurusd: chart

eurusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'eurusd']))
eurusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'eurusd'][['fx_log']])
eurusd_fair['predicted_eurusd'] = eurusd_pred
eurusd_fair.columns = ['actual_eurusd', 'fair_eurusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('eurusd: actual level and long-term fair value (beer)')
eurusd_fair.actual_eurusd.plot(ax=ax[0], label='actual eurusd', legend=True)
eurusd_fair.fair_eurusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = eurusd_fair.actual_eurusd.div(eurusd_fair.fair_eurusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('eurusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('eurusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for cadusd: chart

cadusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'cadusd']))
cadusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'cadusd'][['fx_log']])
cadusd_fair['predicted_cadusd'] = cadusd_pred
cadusd_fair.columns = ['actual_cadusd', 'fair_cadusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('cadusd: actual level and long-term fair value (beer)')
cadusd_fair.actual_cadusd.plot(ax=ax[0], label='actual cadusd', legend=True)
cadusd_fair.fair_cadusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = cadusd_fair.actual_cadusd.div(cadusd_fair.fair_cadusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('cadusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('cadusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for jpyusd: chart

jpyusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'jpyusd']))
jpyusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'jpyusd'][['fx_log']])
jpyusd_fair['predicted_jpyusd'] = jpyusd_pred
jpyusd_fair.columns = ['actual_jpyusd', 'fair_jpyusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('jpyusd: actual level and long-term fair value (beer)')
jpyusd_fair.actual_jpyusd.plot(ax=ax[0], label='actual jpyusd', legend=True)
jpyusd_fair.fair_jpyusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = jpyusd_fair.actual_jpyusd.div(jpyusd_fair.fair_jpyusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('jpyusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('jpyusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for gbpusd: chart

gbpusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'gbpusd']))
gbpusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'gbpusd'][['fx_log']])
gbpusd_fair['predicted_gbpusd'] = gbpusd_pred
gbpusd_fair.columns = ['actual_gbpusd', 'fair_gbpusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('gbpusd: actual level and long-term fair value (beer)')
gbpusd_fair.actual_gbpusd.plot(ax=ax[0], label='actual gbpusd', legend=True)
gbpusd_fair.fair_gbpusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = gbpusd_fair.actual_gbpusd.div(gbpusd_fair.fair_gbpusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('gbpusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('gbpusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for sekusd: chart

sekusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'sekusd']))
sekusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'sekusd'][['fx_log']])
sekusd_fair['predicted_sekusd'] = sekusd_pred
sekusd_fair.columns = ['actual_sekusd', 'fair_sekusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('sekusd: actual level and long-term fair value (beer)')
sekusd_fair.actual_sekusd.plot(ax=ax[0], label='actual sekusd', legend=True)
sekusd_fair.fair_sekusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = sekusd_fair.actual_sekusd.div(sekusd_fair.fair_sekusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('sekusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('sekusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for nokusd: chart

nokusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'nokusd']))
nokusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'nokusd'][['fx_log']])
nokusd_fair['predicted_nokusd'] = nokusd_pred
nokusd_fair.columns = ['actual_nokusd', 'fair_nokusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('nokusd: actual level and long-term fair value (beer)')
nokusd_fair.actual_nokusd.plot(ax=ax[0], label='actual nokusd', legend=True)
nokusd_fair.fair_nokusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = nokusd_fair.actual_nokusd.div(nokusd_fair.fair_nokusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('nokusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('nokusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for chfusd: chart

chfusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'chfusd']))
chfusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'chfusd'][['fx_log']])
chfusd_fair['predicted_chfusd'] = chfusd_pred
chfusd_fair.columns = ['actual_chfusd', 'fair_chfusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('chfusd: actual level and long-term fair value (beer)')
chfusd_fair.actual_chfusd.plot(ax=ax[0], label='actual chfusd', legend=True)
chfusd_fair.fair_chfusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = chfusd_fair.actual_chfusd.div(chfusd_fair.fair_chfusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('chfusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('chfusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for audusd: chart

audusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'audusd']))
audusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'audusd'][['fx_log']])
audusd_fair['predicted_audusd'] = audusd_pred
audusd_fair.columns = ['actual_audusd', 'fair_audusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('audusd: actual level and long-term fair value (beer)')
audusd_fair.actual_audusd.plot(ax=ax[0], label='actual audusd', legend=True)
audusd_fair.fair_audusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = audusd_fair.actual_audusd.div(audusd_fair.fair_audusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('audusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('audusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for nzdusd: chart

nzdusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'nzdusd']))
nzdusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'nzdusd'][['fx_log']])
nzdusd_fair['predicted_nzdusd'] = nzdusd_pred
nzdusd_fair.columns = ['actual_nzdusd', 'fair_nzdusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('nzdusd: actual level and long-term fair value (beer)')
nzdusd_fair.actual_nzdusd.plot(ax=ax[0], label='actual nzdusd', legend=True)
nzdusd_fair.fair_nzdusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = nzdusd_fair.actual_nzdusd.div(nzdusd_fair.fair_nzdusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('nzdusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('nzdusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for plnusd: chart

plnusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'plnusd']))
plnusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'plnusd'][['fx_log']])
plnusd_fair['predicted_plnusd'] = plnusd_pred
plnusd_fair.columns = ['actual_plnusd', 'fair_plnusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('plnusd: actual level and long-term fair value (beer)')
plnusd_fair.actual_plnusd.plot(ax=ax[0], label='actual plnusd', legend=True)
plnusd_fair.fair_plnusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = plnusd_fair.actual_plnusd.div(plnusd_fair.fair_plnusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('plnusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('plnusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for hufusd: chart

hufusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'hufusd']))
hufusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'hufusd'][['fx_log']])
hufusd_fair['predicted_hufusd'] = hufusd_pred
hufusd_fair.columns = ['actual_hufusd', 'fair_hufusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('hufusd: actual level and long-term fair value (beer)')
hufusd_fair.actual_hufusd.plot(ax=ax[0], label='actual hufusd', legend=True)
hufusd_fair.fair_hufusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = hufusd_fair.actual_hufusd.div(hufusd_fair.fair_hufusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('hufusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('hufusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()

# %% fair value for czkusd: chart

czkusd_pred = np.exp(in_sample_model.predict(
    panel_data_and_dummies[panel_data_and_dummies.currency == 'czkusd']))
czkusd_fair = np.exp(panel_data_and_dummies[panel_data_and_dummies.currency == 'czkusd'][['fx_log']])
czkusd_fair['predicted_czkusd'] = czkusd_pred
czkusd_fair.columns = ['actual_czkusd', 'fair_czkusd']  # rename cols: actual and in sample fair value (predicted)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0)
fig.suptitle('czkusd: actual level and long-term fair value (beer)')
czkusd_fair.actual_czkusd.plot(ax=ax[0], label='actual czkusd', legend=True)
czkusd_fair.fair_czkusd.plot(ax=ax[0], color='orange', label='fair value (beer model)', legend=True)

deviation_perc = czkusd_fair.actual_czkusd.div(czkusd_fair.fair_czkusd).sub(1).mul(100)
deviation_perc.plot(ax=ax[1])
ax[1].set_title('czkusd: % deviation versus long-term fair value (beer) (-/+)')
ax[1].set_ylabel('% deviation (-/+)')
ax[1].axhline(0, color='red', linestyle='--')

deviation_z = stats.zscore(deviation_perc)
deviation_z.plot(ax=ax[2])
ax[2].set_title('czkusd: z-score deviation versus long-term fair value (beer) (-/+)')
ax[2].set_ylabel('z-score deviation (-/+)')
ax[2].axhline(0, color='red', linestyle='--')

sns.despine()
plt.show()
