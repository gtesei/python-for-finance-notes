
# coding: utf-8

# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>

# # Python for Finance

# **Analyze Big Financial Data**
# 
# O'Reilly (2014)
# 
# Yves Hilpisch

# <img style="border:0px solid grey;" src="http://hilpisch.com/python_for_finance.png" alt="Python for Finance" width="30%" align="left" border="0">

# **Buy the book ** |
# <a href='http://shop.oreilly.com/product/0636920032441.do' target='_blank'>O'Reilly</a> |
# <a href='http://www.amazon.com/Yves-Hilpisch/e/B00JCYHHJM' target='_blank'>Amazon</a>
# 
# **All book codes & IPYNBs** |
# <a href="http://oreilly.quant-platform.com">http://oreilly.quant-platform.com</a>
# 
# **The Python Quants GmbH** | <a href='http://pythonquants.com' target='_blank'>www.pythonquants.com</a>
# 
# **Contact us** | <a href='mailto:analytics@pythonquants.com'>analytics@pythonquants.com</a>

# # Statistics

# ## Normality Tests

# ### Benchmark Case

# In[1]:

import numpy as np
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

def gen_paths(S0, r, sigma, T, M, I):
    ''' Generate Monte Carlo paths for geometric Brownian motion.
    
    Parameters
    ==========
    S0 : float
        initial stock/index value
    r : float
        constant short rate
    sigma : float
        constant volatility
    T : float
        final time horizon
    M : int
        number of time steps/intervals
    I : int
        number of paths to be simulated
        
    Returns
    =======
    paths : ndarray, shape (M + 1, I)
        simulated paths given the parameters
    '''
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths


# In[3]:

S0 = 100.
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000


# In[4]:

paths = gen_paths(S0, r, sigma, T, M, I)


# In[5]:

plt.plot(paths[:, :10])
plt.grid(True)
plt.xlabel('time steps')
plt.ylabel('index level')
# tag: normal_sim_1
# title: 10 simulated paths of geometric Brownian motion


# In[6]:

log_returns = np.log(paths[1:] / paths[0:-1]) 


# In[7]:

paths[:, 0].round(4)


# In[8]:

log_returns[:, 0].round(4)


# In[9]:

def print_statistics(array):
    ''' Prints selected statistics.
    
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    sta = scs.describe(array)
    print "%14s %15s" % ('statistic', 'value')
    print 30 * "-"
    print "%14s %15.5f" % ('size', sta[0])
    print "%14s %15.5f" % ('min', sta[1][0])
    print "%14s %15.5f" % ('max', sta[1][1])
    print "%14s %15.5f" % ('mean', sta[2])
    print "%14s %15.5f" % ('std', np.sqrt(sta[3]))
    print "%14s %15.5f" % ('skew', sta[4])
    print "%14s %15.5f" % ('kurtosis', sta[5])


# In[10]:

print_statistics(log_returns.flatten())


# In[11]:

plt.hist(log_returns.flatten(), bins=70, normed=True, label='frequency')
plt.grid(True)
plt.xlabel('log-return')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)),
         'r', lw=2.0, label='pdf')
plt.legend()
# tag: normal_sim_2
# title: Histogram of log-returns and normal density function


# In[12]:

sm.qqplot(log_returns.flatten()[::500], line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: sim_val_qq_1
# title: Quantile-quantile plot for log returns


# In[13]:

def normality_tests(arr):
    ''' Tests for normality distribution of given data set.
    
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    print "Skew of data set  %14.3f" % scs.skew(arr)
    print "Skew test p-value %14.3f" % scs.skewtest(arr)[1]
    print "Kurt of data set  %14.3f" % scs.kurtosis(arr)
    print "Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1]
    print "Norm test p-value %14.3f" % scs.normaltest(arr)[1]


# In[14]:

normality_tests(log_returns.flatten())


# In[15]:

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.hist(paths[-1], bins=30)
ax1.grid(True)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.set_title('regular data')
ax2.hist(np.log(paths[-1]), bins=30)
ax2.grid(True)
ax2.set_xlabel('log index level')
ax2.set_title('log data')
# tag: normal_sim_3
# title: Histogram of simulated end-of-period index levels
# size: 90


# In[16]:

print_statistics(paths[-1])


# In[17]:

print_statistics(np.log(paths[-1]))


# In[18]:

normality_tests(np.log(paths[-1]))


# In[19]:

log_data = np.log(paths[-1])
plt.hist(log_data, bins=70, normed=True, label='observed')
plt.grid(True)
plt.xlabel('index levels')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, log_data.mean(), log_data.std()),
         'r', lw=2.0, label='pdf')
plt.legend()
# tag: normal_sim_4
# title: Histogram of log index levels and normal density function


# In[20]:

sm.qqplot(log_data, line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: sim_val_qq_2
# title: Quantile-quantile plot for log index levels


# ### Real World Data

# In[21]:

import pandas as pd
import pandas.io.data as web


# In[22]:

symbols = ['^GDAXI', '^GSPC', 'YHOO', 'MSFT']


# In[23]:

data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo',
                            start='1/1/2006')['Adj Close']
data = data.dropna()


# In[24]:

data.info()


# In[25]:

data.head()


# In[26]:

(data / data.ix[0] * 100).plot(figsize=(8, 6), grid=True)
# tag: real_returns_1
# title: Evolution of stock and index levels over time


# In[27]:

log_returns = np.log(data / data.shift(1))
log_returns.head()


# In[28]:

log_returns.hist(bins=50, figsize=(9, 6))
# tag: real_returns_2
# title: Histogram of respective log-returns
# size: 90


# In[29]:

for sym in symbols:
    print "\nResults for symbol %s" % sym
    print 30 * "-"
    log_data = np.array(log_returns[sym].dropna())
    print_statistics(log_data)


# In[30]:

sm.qqplot(log_returns['^GSPC'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: real_val_qq_1
# title: Quantile-quantile plot for S&P 500 log returns


# In[31]:

sm.qqplot(log_returns['MSFT'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: real_val_qq_2
# title: Quantile-quantile plot for Microsoft log returns


# In[32]:

for sym in symbols:
    print "\nResults for symbol %s" % sym
    print 32 * "-"
    log_data = np.array(log_returns[sym].dropna())
    normality_tests(log_data)


# ## Portfolio Optimization

# ### The Data

# In[33]:

import numpy as np
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[34]:

symbols = ['AAPL', 'MSFT', 'YHOO', 'DB', 'GLD']
noa = len(symbols)


# In[35]:

data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo',
                               end='2014-09-12')['Adj Close']
data.columns = symbols


# In[36]:

(data / data.ix[0] * 100).plot(figsize=(8, 5), grid=True)
# tag: portfolio_1
# title: Stock prices over time
# size: 90


# In[37]:

rets = np.log(data / data.shift(1))


# In[38]:

rets.mean() * 252


# In[39]:

rets.cov() * 252


# ### The Basic Theory

# In[40]:

weights = np.random.random(noa)
weights /= np.sum(weights)


# In[41]:

weights


# In[42]:

np.sum(rets.mean() * weights) * 252
  # expected portfolio return


# In[43]:

np.dot(weights.T, np.dot(rets.cov() * 252, weights))
  # expected portfolio variance


# In[44]:

np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
  # expected portfolio standard deviation/volatility


# In[45]:

prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T, 
                        np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)


# In[46]:

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
# tag: portfolio_2
# title: Expected return and volatility for different/random portfolio weights
# size: 90


# ### Portfolio Optimizations

# In[47]:

def statistics(weights):
    ''' Return portfolio statistics.
    
    Parameters
    ==========
    weights : array-like
        weights for different securities in portfolio
    
    Returns
    =======
    pret : float
        expected portfolio return
    pvol : float
        expected portfolio volatility
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])


# In[48]:

import scipy.optimize as sco


# In[49]:

def min_func_sharpe(weights):
    return -statistics(weights)[2]


# In[50]:

cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})


# In[51]:

bnds = tuple((0, 1) for x in range(noa))


# In[52]:

noa * [1. / noa,]


# In[53]:

get_ipython().run_cell_magic('time', '', "opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',\n                       bounds=bnds, constraints=cons)")


# In[54]:

opts


# In[55]:

opts['x'].round(3)


# In[56]:

statistics(opts['x']).round(3)


# In[57]:

def min_func_variance(weights):
    return statistics(weights)[1] ** 2


# In[58]:

optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)


# In[59]:

optv


# In[60]:

optv['x'].round(3)


# In[61]:

statistics(optv['x']).round(3)


# ### Efficient Frontier

# In[62]:

cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in weights)


# In[63]:

def min_func_port(weights):
    return statistics(weights)[1]


# In[64]:

get_ipython().run_cell_magic('time', '', "trets = np.linspace(0.0, 0.25, 50)\ntvols = []\nfor tret in trets:\n    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},\n            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})\n    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',\n                       bounds=bnds, constraints=cons)\n    tvols.append(res['fun'])\ntvols = np.array(tvols)")


# In[65]:

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets,
            c=prets / pvols, marker='o')
            # random portfolio composition
plt.scatter(tvols, trets,
            c=trets / tvols, marker='x')
            # efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
            # portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
            # minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
# tag: portfolio_3
# title: Minimum risk portfolios for given return level (crosses)
# size: 90


# ### Capital Market Line

# In[66]:

import scipy.interpolate as sci


# In[67]:

ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]


# In[68]:

tck = sci.splrep(evols, erets)


# In[69]:

def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der=0)
def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)


# In[70]:

def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


# In[71]:

opt = sco.fsolve(equations, [0.01, 0.5, 0.15])


# In[72]:

opt


# In[73]:

np.round(equations(opt), 6)


# In[74]:

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets,
            c=(prets - 0.01) / pvols, marker='o')
            # random portfolio composition
plt.plot(evols, erets, 'g', lw=4.0)
            # efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
            # capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0) 
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
# tag: portfolio_4
# title: Capital market line and tangency portfolio (star) for risk-free rate of 1%
# size: 90


# In[75]:

cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)


# In[76]:

res['x'].round(3)


# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>
# 
# <a href="http://www.pythonquants.com" target="_blank">www.pythonquants.com</a> | <a href="http://twitter.com/dyjh" target="_blank">@dyjh</a>
# 
# <a href="mailto:analytics@pythonquants.com">analytics@pythonquants.com</a>
# 
# **Python Quant Platform** |
# <a href="http://oreilly.quant-platform.com">http://oreilly.quant-platform.com</a>
# 
# **Derivatives Analytics with Python** |
# <a href="http://www.derivatives-analytics-with-python.com" target="_blank">Derivatives Analytics @ Wiley Finance</a>
# 
# **Python for Finance** |
# <a href="http://shop.oreilly.com/product/0636920032441.do" target="_blank">Python for Finance @ O'Reilly</a>
