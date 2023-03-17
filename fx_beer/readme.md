# GOAL
Use Python for FX-valuation modelling

Construct a long-term equilibrium fair value model to value G12 FX Currency Pairs: G9 + 3 CEE-currencies: eurusd, usdcad, usdjpy, gbpusd, usdsek, usdnok, usdchf, audusd, nzdusd, usdpln, usdhuf, usdczk. Run a in-sample Fixed Effect Panel Regression. Determine currencies with largest deviations in valuation. 

Construct an Error Correction Model to estimate the impact of valuation misalignment on future movements in the actual exchange rate.

Perform out-of-sample predictive, recursive panel regression.

Construct a macro trading model based takes long-term positions in currencies only if the currency misalignment versus the fair value estimate is extreme.
![image](https://user-images.githubusercontent.com/106360966/226069400-f592e0d9-512e-47e4-b34d-09cb3c7e1ba6.png)
