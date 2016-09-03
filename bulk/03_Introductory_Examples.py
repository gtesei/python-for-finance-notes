
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

# # Introductory Examples

# In[1]:

import warnings
warnings.simplefilter('ignore')


# ## Implied Volatilities

# In[2]:

V0 = 17.6639


# In[3]:

r = 0.01


# In[4]:

import pandas as pd
h5 = pd.HDFStore('./source/vstoxx_data_31032014.h5', 'r')
futures_data = h5['futures_data']  # VSTOXX futures data
options_data = h5['options_data']  # VSTOXX call option data
h5.close()


# In[5]:

futures_data


# In[6]:

options_data.info()


# In[7]:

options_data[['DATE', 'MATURITY', 'TTM', 'STRIKE', 'PRICE']].head()


# In[8]:

options_data['IMP_VOL'] = 0.0
  # new column for implied volatilities


# In[9]:

from bsm_functions import *


# In[10]:

tol = 0.5  # tolerance level for moneyness
for option in options_data.index:
    # iterating over all option quotes
    forward = futures_data[futures_data['MATURITY'] ==                 options_data.loc[option]['MATURITY']]['PRICE'].values[0]
      # picking the right futures value
    if (forward * (1 - tol) < options_data.loc[option]['STRIKE']
                             < forward * (1 + tol)):
        # only for options with moneyness within tolerance
        imp_vol = bsm_call_imp_vol(
                V0,  # VSTOXX value 
                options_data.loc[option]['STRIKE'],
                options_data.loc[option]['TTM'],
                r,   # short rate
                options_data.loc[option]['PRICE'],
                sigma_est=2.,  # estimate for implied volatility
                it=100)
        options_data['IMP_VOL'].loc[option] = imp_vol


# In[11]:

futures_data['MATURITY']
  # select the column with name MATURITY


# In[12]:

options_data.loc[46170]
  # select data row for index 46170


# In[13]:

options_data.loc[46170]['STRIKE']
  # select only the value in column STRIKE
  # for index 46170 


# In[14]:

plot_data = options_data[options_data['IMP_VOL'] > 0]


# In[15]:

maturities = sorted(set(options_data['MATURITY']))
maturities


# In[16]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.figure(figsize=(8, 6))
for maturity in maturities:
    data = plot_data[options_data.MATURITY == maturity]
      # select data for this maturity
    plt.plot(data['STRIKE'], data['IMP_VOL'],
             label=maturity.date(), lw=1.5)
    plt.plot(data['STRIKE'], data['IMP_VOL'], 'r.')
plt.grid(True) 
plt.xlabel('strike')
plt.ylabel('implied volatility of volatility')
plt.legend()
plt.show()
# tag: vs_imp_vol
# title: Implied volatilities (of volatility) for European call options on the VSTOXX on 31. March 2014


# In[17]:

keep = ['PRICE', 'IMP_VOL']
group_data = plot_data.groupby(['MATURITY', 'STRIKE'])[keep]
group_data


# In[18]:

group_data = group_data.sum()
group_data.head()


# In[19]:

group_data.index.levels


# ## Monte Carlo Simulation

# In[20]:

from bsm_functions import bsm_call_value
S0 = 100.
K = 105.
T = 1.0
r = 0.05
sigma = 0.2
bsm_call_value(S0, K, T, r, sigma)


# ### Pure Python

# In[21]:

get_ipython().magic('run mcs_pure_python.py')


# In[22]:

sum_val = 0.0
for path in S:
    # C-like iteration for comparison
    sum_val += max(path[-1] - K, 0)
C0 = exp(-r * T) * sum_val / I
round(C0, 3)


# ### Vectorization with NumPy

# In[23]:

v = range(1, 6)
print v


# In[24]:

2 * v


# In[25]:

import numpy as np
v = np.arange(1, 6)
v


# In[26]:

2 * v


# In[27]:

get_ipython().magic('run mcs_vector_numpy.py')


# In[28]:

round(tpy / tnp1, 2)


# ### Full Vectorization with Log Euler Scheme

# In[29]:

get_ipython().magic('run mcs_full_vector_numpy.py')


# ### Graphical Analysis

# In[30]:

import matplotlib.pyplot as plt
plt.plot(S[:, :10])
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('index level')
# tag: index_paths
# title: The first 10 simulated index level paths


# In[31]:

plt.hist(S[-1], bins=50)
plt.grid(True)
plt.xlabel('index level')
plt.ylabel('frequency')
# tag: index_histo
# title: Histogram of all simulated end of period index level values


# In[32]:

plt.hist(np.maximum(S[-1] - K, 0), bins=50)
plt.grid(True)
plt.xlabel('option inner value')
plt.ylabel('frequency')
plt.ylim(0, 50000)
# tag: option_iv_hist
# title: Histogram of all simulated end of period option inner values


# In[33]:

sum(S[-1] < K)


# ## Technical Analysis

# In[34]:

import numpy as np
import pandas as pd
import pandas.io.data as web


# In[35]:

sp500 = web.DataReader('^GSPC', data_source='yahoo',
                       start='1/1/2000', end='4/14/2014')
sp500.info()


# In[36]:

sp500['Close'].plot(grid=True, figsize=(8, 5))
# tag: sp500
# title: Historical levels of the S&P 500 index


# In[37]:

sp500['42d'] = np.round(pd.rolling_mean(sp500['Close'], window=42), 2)
sp500['252d'] = np.round(pd.rolling_mean(sp500['Close'], window=252), 2)


# In[38]:

sp500[['Close', '42d', '252d']].tail()


# In[39]:

sp500[['Close', '42d', '252d']].plot(grid=True, figsize=(8, 5))
# tag: sp500_trend
# title: The S&P 500 index with 42d and 252d trend lines


# In[40]:

sp500['42-252'] = sp500['42d'] - sp500['252d']
sp500['42-252'].tail()


# In[41]:

sp500['42-252'].head()


# In[42]:

SD = 50
sp500['Regime'] = np.where(sp500['42-252'] > SD, 1, 0)
sp500['Regime'] = np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])
sp500['Regime'].value_counts()


# In[43]:

sp500['Regime'].plot(lw=1.5, grid=True)
plt.ylim([-1.1, 1.1])
# tag: sp500_signal
# title: Signal regimes over time


# In[44]:

sp500['Market'] = np.log(sp500['Close'] / sp500['Close'].shift(1))


# In[45]:

sp500['Strategy'] = sp500['Regime'].shift(1) * sp500['Market']


# In[46]:

sp500[['Market', 'Strategy']].cumsum().apply(np.exp).plot(grid=True,
                                                    figsize=(8, 5))
# tag: sp500_wealth
# title: The S&P 500 index vs. investor's wealth


# ## Conclusions

# ## Further Reading

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
