# GOAL
Use Python for FX-valuation modelling

Construct a long-term equilibrium fair value model to value G12 FX Currency Pairs: G9 + 3 CEE-currencies: eurusd, usdcad, usdjpy, gbpusd, usdsek, usdnok, usdchf, audusd, nzdusd, usdpln, usdhuf, usdczk. Run a in-sample Fixed Effect Panel Regression. Determine currencies with largest deviations in valuation. 

Construct an Error Correction Model to estimate the impact of valuation misalignment on future movements in the actual exchange rate.

Perform out-of-sample predictive, recursive panel regression.

Construct a macro trading model based takes long-term positions in currencies only if the currency misalignment versus the fair value estimate is extreme.

# FILES
main.py: the source code of the project <br>
requirements.txt: required packages to run main.py <br>
fx_beer_data_xlsx: excel file contains the data used in main (fx levels, terms of trade, gross fixed capital formation % gdp, <br>
yield levels, cpi index, productivity index) <br>
g12_fx_valuation.ppt: powerpoint presentation with our findings <br>
