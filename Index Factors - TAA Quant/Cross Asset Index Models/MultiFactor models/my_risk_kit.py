import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.stats
import math

def expanding_z_score(seq, warmup=24):
    """ Returns a Series or  DataFrame of z-scores calculated on an expanding window with a warmup period.
    
    Args:
        seq(Series or DataFrame): a sequence of values for which a z-score will be calculated on the expanding window.
        warmup(int): The warm-up period is the first period that is used to calculate the first z-score. After that the 
            z-score is calculated on an expanding window of the sequence: from the very first start value until the next step.
    Returns:
        Series or DataFrame: z-scores calculated on on an expanding window of values.
    """
    average_expanding=seq.expanding(min_periods=warmup).mean() #expanding mean
    std_expanding=seq.expanding(min_periods=warmup).std() #expanding stdev
    z_expanding=(seq-average_expanding)/std_expanding #the expanding z-score
    return z_expanding

def rolling_z_score(seq, rol_window=24):
    """ Returns a Series or  DataFrame of z-scores calculated on a rolling window.
    
    Args:
        seq(Series or DataFrame): a sequence of values for which a  z-score will be calculated on the rolling window.
        rol_window(int): The rolling window period.
        
    Returns:
        Series or DataFrame: A sequence of z-scores calculated on a rolling window of values.
    """
    average_rolling=seq.rolling(window=rol_window).mean() #rolling mean
    std_rolling=seq.rolling(window=rol_window).std() #rolling stdev
    z_rolling=(seq-average_rolling)/std_rolling #the rolling z-score
    return z_rolling

def trim_z_scores(seq,lower=-3,upper=+3):
    """Trims z-score of Series or DataFrame to specified values"""
    
    
    def replace(element,lower,upper):
        if element < lower:
            element = lower
        elif element > upper:
            element = upper
        else:
            element=element
        return element
      
    
    if isinstance(seq,pd.Series):
        seq.loc[seq < lower]=lower
        seq.loc[seq > upper]=upper
    else:
        seq=seq.applymap(replace,lower=lower,upper=upper)
                
    return seq




























def mom_score(r,per=12,composite=False):
    """"Calculates a momentum score on a Series or DataFrame of returns.
    
    Computes a simple momentum-score for one series of returns or for several return series 
    (several assets) in a DataFrame
    
    Args:
        r(Series or DataFrame): The returns.
        per(int):The momentum lookback_period.
        composite(bool): default is False in which case the function uses the per parameter.
        
    Returns:
        Series or DataFrame:The composite momentum score for each period (for each asset)
     """
    
    mom=(r+1).rolling(per).apply(np.prod)-1
    if composite:
        ret1M=r
        ret3M=(r+1).rolling(3).apply(np.prod)-1
        ret6M=(r+1).rolling(6).apply(np.prod)-1
        ret12M=(r+1).rolling(12).apply(np.prod)-1
        mom=0.25*ret1M+0.25*ret3M+0.25*ret6M+0.25*ret12M
    return mom


#FUNCTIONS FOR ABSOLUTE PERFORMANCE

def annualized_return(r,per):
    """ Annualized return calculated from a pd.Series or DataFrame.r may contain null values at start series.
    Args:
        r(Series or DataFrame):return per period in R-format (eg:0.005=0.5%)
        per(int): the period adjustment (12 for monthly periods, 52 for weekly periods, 250 for daily periods)
    Returns:
        Float or Series: annualized return
    """
    yearfraq=(r.shape[0]-r.isna().sum())/per
    wealth=(1+r).prod()
    annual_return=wealth**(1/yearfraq)-1
    return annual_return  

def annualized_vol(r,per):
    """ Annualized volatility calculated from a return series.
    Args:
        r(pd.Series):return series per period in R-format (e.g: 0.005 = 0.5%)
        per(int): the period adjustment (12 for monthly periods, 52 for weekly periods, 250 for daily periods)
    Returns:
        Float: annualized volatility
    """
    annual_vol=r.dropna().std()*np.sqrt(per)
    return annual_vol

def downside_vol(r,per):
    """ Annualized volatility of the negative returns calculated from a return series.
    Args:
        r(pd.Series):return series per period in R-format (e.g: 0.005 = 0.5%)
        per(int): the period adjustment (12 for monthly periods, 52 for weekly periods, 250 for daily periods)
    Returns:
        Float: annualized volatility
    """
    down_vol=r.dropna()[r<0].std()*np.sqrt(per)
    return down_vol

def max_drawdown(r):
    """ Calculates the maximum drawdown over total period of a return series
    Args:
        r(pd.Series or DataFrame):the return series
    Returns:
        Float: maximum drawdown
    """
    wealth_index=(1+r).cumprod()
    max_wealth=wealth_index.cummax()
    drawdown_index=wealth_index/max_wealth-1
    max_dd=drawdown_index.min()
    return max_dd

def drawdown_series(r):
    """ Calculates the rolling drawdown of a return series
    Args:
        r(pd.Series):the return series in R format (e.g.: 5% = 0.05)
    Returns:
        pd.Series: rolling drawdown
    """
    wealth_index=(1+r).cumprod()
    max_wealth=wealth_index.cummax()
    drawdown_index=wealth_index/max_wealth-1
    return drawdown_index

def sharpe_ratio(r1,r2,per):
    """ Sharpe ratio of a strategy versus risk-free rate based on return series input.
    Args:
        r1(pd.Series):return series of strategy per period in R-format (e.g: 0.005 = 0.5%)
        r2:return series of risk-free asset per period in R-format
        per(int): the period adjustment (12 for monthly periods, 52 for weekly periods, 250 for daily periods)
    Returns:
        Float: sharpe ratio
    """
    strategy_return=annualized_return(r1,per)
    risk_free_return=annualized_return(r2,per)
    strategy_volatility=annualized_vol(r1,per)
    sharpe=(strategy_return-risk_free_return)/strategy_volatility
    
    return sharpe

def sortino_ratio(r1,r2,per):
    """ Sortino ratio of a strategy versus risk-free rate based on return series input.
    Args:
        r1(pd.Series):return series of strategy per period in R-format (e.g: 0.005 = 0.5%)
        r2:return series of risk-free asset per period in R-format
        per(int): the period adjustment (12 for monthly periods, 52 for weekly periods, 250 for daily periods)
    Returns:
        Float: sharpe ratio
    """
    strategy_return=annualized_return(r1,per)
    risk_free_return=annualized_return(r2,per)
    strategy_downside_volatility=downside_vol(r1,per)
    sortino=(strategy_return-risk_free_return)/strategy_downside_volatility
    
    return sortino


def mar_ratio(r,per=12):
    """ MAR ratio of a strategy: annulized return divided by maximum drawdown for a return series
    Args:
        r(pd.Series):return series of strategy per period in R-format (e.g: 0.005 = 0.5%)
        per: period adjustement for frequency period of series.default is 12 for monthly frequency (52 for weeks, 250 for days).
            
    Returns:
        Float: MAR-ratio
    """
    r_annualized_return=annualized_return(r,per)
    r_max_drawdown=max_drawdown(r)
    mar=r_annualized_return/abs(r_max_drawdown)
    
    return mar


def best_periods(r):
    """ Maximum return, maximum annual return and % of positive returns in a return time-series.
    Args:
        r(pd.Series or Dataframe):return per period in R-format (eg:0.005=0.5%)
    Returns:
        Tuple: tuple with 3 elements as floats.
    """
    best_period_return=r.max()
    annual_return_series=(1+r).dropna().resample('A').prod()-1
    best_annual_return=annual_return_series.max()
    profitable_periods=(r.dropna()>0).mean()
    
    return (best_period_return,best_annual_return,profitable_periods)

def worst_periods(r):
    """ Minimum return,  minimum annual return and % of negative returns in a return time-series.
    Args:
        r(pd.Series or Dataframe):return per period in R-format (eg:0.005=0.5%)
    Returns:
        Tuple: tuple with 3 elements as floats.
    """
    worst_period_return=r.min()
    annual_return_series=(1+r).dropna().resample('A').prod()-1
    worst_annual_return=annual_return_series.min()
    losing_periods=(r.dropna()<0).mean()
    
    return (worst_period_return,worst_annual_return,losing_periods)


def succes_ratio_absolute(r,years=3,per=12):
    """ Percentage of positive rolling period returns in a return series.
    Args:
        r(pd.Series or Dataframe):return per period in R-format (eg:0.005=0.5%)
        years(int): number of years to calculate the success ratio for. Default is 3 years.
        per(int): period adjustment for the frequency period of the series. Default is 12. (52 for weeks,250 for days)
    Returns:
        float: succes ratio
    """
    rolling_return_series=(1+r.dropna()).rolling(years*per).apply(np.prod)-1
    base_rate_absolute=(rolling_return_series.dropna()>0).mean()
     
    return base_rate_absolute

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns kurtosis as a float or a Series. Scipy function returns excess kurtosis above 3.
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def historic_var(r,level=0.01):
    """
    Value at risk for the frequency period of the time-series.
    Args:
    r(pd.Series or Df): the return series in R-format (e.g. 5% = 0.05)
    level(float): the significance level,the quantile(percentile) in the return series where level% lies below.
    """
    hist_var=np.quantile(r.dropna(),q=level)
  
    return hist_var


def historic_cvar(r,level=0.01):
    """
    Conditional Value at risk for the frequency period of the time-series.
    Args:
    r(pd.Series or Df): the return series in R-format (e.g. 5% = 0.05)
    level(float): the significance level,the quantile(percentile) in the return series where level% lies below.
    """
    cvar_mask=r.dropna()<historic_var(r)
    hist_cvar=r.dropna()[cvar_mask].mean()
  
    return hist_cvar


def var_gaussian(r, level=1, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return r.mean() + z*r.std(ddof=0)


#FUNCTIONS FOR RELATIVE PERFORMANCE VERSUS BENCHMARK


def information_ratio(r1,r2,per):
    """ Calculates the information ratio of a strategy based on return series strategy, return series benchmark 
    and period adjustment factor to annualized volatlity.
    ir= (annual return strategy - annual return bench)/tracking error
    Args:
        r1(pd.Series):first return series per period in R-format (e.g: 0.005 = 0.5%)
        r2(pd.Series):second return series per period in R-format
        per(int): the period adjustment (12 for monthly periods, 52 for weekly periods, 250 for daily periods)
    Returns:
        Float: information ratio
    """
    annual_return_strategy=annualized_return(r1,per)
    annual_return_benchmark=annualized_return(r2,per)
    track_error=tracking_error(r1,r2,per)
    ir = (annual_return_strategy-annual_return_benchmark)/track_error
    return ir


def tracking_error(r1,r2,per):
    """ Annualized tracking error calculated from 2 return series.
    Args:
        r1(pd.Series):first return series per period in R-format (e.g: 0.005 = 0.5%)
        r2(pd.Series):second return series per period in R-format
        per(int): the period adjustment (12 for monthly periods, 52 for weekly periods, 250 for daily periods)
    Returns:
        Float: annualized tracking error
    """
    excess_ret=r1-r2
    track_error=excess_ret.std()*np.sqrt(12)
    return track_error


def relative_drawdown_series(r1,r2):
    """ relative drawdown of a return series versus another benchmark return series"""
    relative_excess_returns=(r1-r2).dropna()
    relative_wealth_index=(1+relative_excess_returns).cumprod()
    max_relative_wealth_index=relative_wealth_index.cummax()
    rel_dd_series=relative_wealth_index/max_relative_wealth_index-1
    return rel_dd_series




def succes_ratio_relative(r1,r2,years=3,per=12):
    """ Percentage of positive rolling period excess returns in a return series versus another return series (benchmark.)
    Args:
        r1(pd.Series or Dataframe):return per period in R-format (eg:0.005=0.5%).
        r2(pd.Series or Dataframe):return per period in R-format (eg:0.005=0.5%), the benchmark to 
        calculate excess returns from.
        years(int): number of years to calculate the success ratio for. Default is 3 years.
        per(int): period adjustment for the frequency period of the series. Default is 12 for months.
        (52 for weeks,250 for days).
    Returns:
        float: relative succes ratio of the excess returns
    """
    rolling_return_series_1=((1+r1.dropna()).rolling(years*per).apply(np.prod)-1).dropna()
    rolling_return_series_2=((1+r2.dropna()).rolling(years*per).apply(np.prod)-1).dropna()
    rolling_excess_return_series=(rolling_return_series_1-rolling_return_series_2).dropna()
    base_rate_relative=(rolling_excess_return_series>0).mean()
     
    return base_rate_relative

#FUNCTION FOR ABSOLUTE PERFORMANCE METRICS DF

def performance_metrics_absolute(name,r,r_rf,per=12):
    """"Dataframe with performance metrics for returns series as input.
    Args:
        name(str): name of return series being analyzed
        r(pd.Series): return series
        r_rf(pd.Series):return series of risk-free instrument
        per(int): period adjustment. default is 12 for monthly frequency.
    Returns:
        DateFrame: DataFrame with absolute performance metrics for the return series.
    """
    
    metrics_names=['CAGR %','Volatility %', 'Downside volatility %','Max drawdown %','Sharpe ratio','Sortino ratio', 
                   'MAR ratio','','Best month %','Worst month %','Best year %', 'Worst year %', 'months profitable %',
                  'succes ratio: rolling 1-year %','succes ratio: rolling 5-year','succes ratio: rolling 10-year','',
                   'Skew','Kurtosis','Historic monthly VAR(1%) %','Gaussian monthly VAR(1%)','Historic monthly CVAR(1%) %'
                  ]
    
      
    metrics_values=[
        round(annualized_return(r,per)*100,2),round(annualized_vol(r,per)*100,2),round(downside_vol(r,per)*100,2),
        round(max_drawdown(r)*100,2),round(sharpe_ratio(r,r_rf,per),2),round(sortino_ratio(r,r_rf,per),2),
        round(mar_ratio(r,per),2),'',round(best_periods(r)[0]*100,2),round(worst_periods(r)[0]*100,2),
        round(best_periods(r)[1]*100,2),round(worst_periods(r)[1]*100,2),round(best_periods(r)[2]*100,2),
        round(succes_ratio_absolute(r,years=1)*100,2),round(succes_ratio_absolute(r,years=5)*100,2),
        round(succes_ratio_absolute(r,years=10)*100,2),'',round(skewness(r),2),round(kurtosis(r),2),
        round(historic_var(r)*100,2),round(var_gaussian(r)*100,2),round(historic_cvar(r)*100,2),
    ]
    
    metrics_dict={'absolute performance metrics':metrics_names,f'{name}':metrics_values}
    
    metrics_df=pd.DataFrame(metrics_dict)
    metrics_df=pd.DataFrame(metrics_dict)
    metrics_df.set_index('absolute performance metrics',inplace=True)
    
    return metrics_df

#FUNCTION FOR RELATIVE PERFORMANCE METRICS DF

def performance_metrics_relative(name1,name2,r1,r2,per=12):
    """"Dataframe with relative performance metrics for returns series as input.
    Args:
        name(str): name of the return series r1 being analyzed
        r1(pd.Series): return series
        r2(pd.Series):return series that will act as the benchmark to compare against
        per(int): period adjustemet. default is 12 for monthly frequency.
    Returns:
        DateFrame: DataFrame with relative performance metrics
    """
    
    metrics_relative_names=['Excess return (CAGR) %','Tracking error %', 'Information ratio',
                  'relative succes ratio: rolling 1-year %','relative succes ratio: rolling 5-year',
                   'relative succes ratio: rolling 10-year',
                  ]
    
      
    metrics_relative_values=[
        round((annualized_return(r1,per)-annualized_return(r2,per))*100,2),round(tracking_error(r1,r2,per)*100,2),
        round(information_ratio(r1,r2,per),2),round(succes_ratio_relative(r1,r2,years=1)*100,2),
        round(succes_ratio_relative(r1,r2,years=5)*100,2),round(succes_ratio_relative(r1,r2,years=10)*100,2)
    ]
    
    metrics_relative_dict={'relative performance metrics':metrics_relative_names,f'{name1}\
    versus {name2}':metrics_relative_values}
    
    metrics_relative_df=pd.DataFrame(metrics_relative_dict)
    metrics_relative_df=pd.DataFrame(metrics_relative_dict)
    metrics_relative_df.set_index('relative performance metrics',inplace=True)
    
    return metrics_relative_df


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level

def portfolio_return(weights, returns):
    """Computes the return on a portfolio.
    
    Computes the portfolio return of n assets given a weights and return vector holding
    the weights and the returns of all n-assets.
    
    Args:
        weights(ndarray or Series): A n*1 ndarray or n*1 series holding the weights(float) of all the n assets.
        returns(ndarray or Series): A n*1 ndarray or n*1 series holding the returns(float) of all the n assets.
    
    Returns:
        Float: The portfolio return of the n-assets
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=".-")


from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    """ Returns the optimal weights that achieves the target return.From target return ==> weight vector.
    
    Function returns the optimal weight vector (holding the optimal weights) that achieve the target return 
    for a portfolio of n assets, given a set of n expected returns and a n*n covmatrix. At the optimal weights 
    the volatility of the portfolio is minimized.
    
    Args:
        target_return(float): The target return to achieve by combining the assets into a portfolio.
        er(ndarrary or Series): The expected return vector which is an n*1 ndarray or n*1 matrix.
        cov(DataFrame): The covariance n*n matrix of the assets.
       
    Returns:
        ndarray: weight vector of size n*1 holding the optimal n weights(floats) that achieve the 
        target return and minimize the volatility of the portfolio of n assets. 
    
    
    """
    
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def optimal_weights(n_points, er, cov):
    """Generates a list of optimal weight vectors that minimize vol for a portfolio of n-assets
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor_history": floorval_history
    }
    return backtest_result


def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    Returns:
        DataFrame: if prices, a DataFrame of n_years_n_steps_per_year rows 
        ndarray: if returns a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def show_gbm(n_scenarios, mu, sigma):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
    s_0=100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False, color="indianred", alpha = 0.5, linewidth=2, figsize=(12,5))
    ax.axhline(y=100, ls=":", color="black")
    # draw a dot at the origin
    ax.plot(0,s_0, marker='o',color='purple', alpha=0.2)
    

import matplotlib.pyplot as plt
import numpy as np

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, steps_per_year=12, y_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = run_cppi(risky_r=pd.DataFrame(risky_r),riskfree_rate=riskfree_rate,m=m, start=start, floor=floor)
    wealth = btr["Wealth"]

    # calculate terminal wealth stats
    y_max=wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]

    
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios

    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0

    # Plot!
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9),xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85),xycoords='axes fraction', fontsize=24)
    if (floor > 0.01):
        hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(.7, .7), xycoords='axes fraction', fontsize=24)

    
def discount(t, r):
    """Compute a discount factor.
    
    The function computes a discount factor or the price of a pure discount bond that
    pays a dollar at time t.
    
    Args:
    
    t(float): Time to maturity in years.
    r(float): Annual interest rate
    
    Returns:
        float: Discount factor or price of zerobond.    
    """
    return (1+r)**(-t)

def pv(l, r):
    """ Compute the present value of a series of liabilities. 
    
    Function computes the present value of a future liability or series of future liabilities given a certain 
    interest rate. The function can also be used to calculate the present value of a zerocoupon bond.
    
    Args:
        l(Series): Series of liabilities defined as pd.Series(data=[amount1,amount2,...],index=[ttm1,ttm2,...]).
            The amount of liabilities are the values of the Series. The Index of the series is the time to maturity(float)
            in years of each liability.
        r(float): Annualized interest rate with annual compounding.
    """
    time_to_maturities = l.index
    discounts = discount(time_to_maturities, r)
    return (discounts*l).sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return assets/pv(liabilities, r)

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.exp(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

import math
def cir(n_years = 10,steps_per_year=12, n_scenarios=1, a=0.05, b=0.03, sigma=0.05,r_0=None):
    """Generate random interest rate evolution over time using the Cox Ingersoll Ross model. 
    
    Args:
        n_years(float): Number of years to model.
        steps_per_year(int): Number of steps per year. Default is monthly steps equal to 12.
        n_scenarios(int): Number of repeated interest rate sequences to model. Number of columns.
        a(float): Speed of mean reversion in the model.
        b(float): Long term average rate annual compounding basis.Not the short rate.
        sigma(float): Annual volatility of if the interest rate.
        r_0(float): Initial starting interest rate on an annual compounding basis. Not the short rate. If None, the long term
        average b is taken.The function takes care of converting this rate to an continuous annual rate.
    Returns:
        DataFrame: DataFrame with rows equal to n_years*steps_per_year and n_scenarios as columns. Each column represents 
        a modelled interest rate with annual compounding (functions converts back the short rates to annual).
    """
    if r_0 is None:
        r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def show_cir_prices(r_0=0.03, a=0.5, b=0.03, sigma=0.05, n_scenarios=5):
    rates,prices= cir(r_0=r_0, a=a, b=b, sigma=sigma, n_scenarios=n_scenarios)
    prices.plot(legend=False, figsize=(12,5))

def show_cir(r_0=0.03, a=0.5, b=0.03, sigma=0.05, n_scenarios=5):
    rates,prices= cir(r_0=r_0, a=a, b=b, sigma=sigma, n_scenarios=n_scenarios)
    rates.plot(legend=False, figsize=(12,5))

#FUNCTIONS FOR DATAFRAME STYLE

def style_df(df_):
    def style_negative(v, props=''):
        return props if v < 0 else None
    
    cell_hover = {
    "selector": "td:hover",
    "props": [("background-color", "#7FB3D5")]
    }
    index_names = {
    "selector": ".index_name",
    "props": "font-style: italic; color: white"
    }
    headers = {
    "selector": "th",
    "props": "background-color: #273746; color: white"
    }
    cells = {
    "selector": "td",
    "props": "background-color: white; font-size: 13px;font-family: sans-serif;border-collapse:collapse; border: 1px solid"
    }  
    x=df_.style\
        .applymap(style_negative, props='color:red;')\
        .format(formatter='{:,.2f}%',na_rep='-')\
        .set_table_styles([cell_hover,index_names, headers,cells])
    
    
def get_fred_yield(per):
    """Get the historical yield for a tenor from Fred database on a monhtly basis. 
    Rename the series.
    Args:
        per(int): Term.Tenor. Year.
    """
    yld=fred.get_series(f'DGS{per}').resample('M').last()
    yld.index.rename('date',inplace=True)
    yld.name=f'yield_{per}y'
    return yld       