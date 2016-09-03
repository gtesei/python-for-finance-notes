
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

# # Volatility Options

# ## The VSTOXX Data

# In[1]:

import sys
sys.path.append('../python')


# In[2]:

import numpy as np
import pandas as pd


# ### VSTOXX Index Data

# In[3]:

url = 'http://www.stoxx.com/download/historical_values/h_vstoxx.txt'
vstoxx_index = pd.read_csv(url, index_col=0, header=2,
                           parse_dates=True, dayfirst=True,
                           sep=',')


# In[4]:

vstoxx_index.info()


# In[5]:

vstoxx_index = vstoxx_index[('2013/12/31' < vstoxx_index.index)
                            & (vstoxx_index.index < '2014/4/1')]


# In[6]:

np.round(vstoxx_index.tail(), 2)


# ### VSTOXX Futures Data

# In[7]:

vstoxx_futures = pd.read_excel('./source/vstoxx_march_2014.xlsx',
                               'vstoxx_futures')


# In[8]:

vstoxx_futures.info()


# In[9]:

del vstoxx_futures['A_SETTLEMENT_PRICE_SCALED']
del vstoxx_futures['A_CALL_PUT_FLAG']
del vstoxx_futures['A_EXERCISE_PRICE']
del vstoxx_futures['A_PRODUCT_ID']


# In[10]:

columns = ['DATE', 'EXP_YEAR', 'EXP_MONTH', 'PRICE']
vstoxx_futures.columns = columns


# In[11]:

import datetime as dt
import calendar

def third_friday(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
    return dt.datetime(date.year, date.month, day)


# In[12]:

set(vstoxx_futures['EXP_MONTH'])


# In[13]:

third_fridays = {}
for month in set(vstoxx_futures['EXP_MONTH']):
    third_fridays[month] = third_friday(dt.datetime(2014, month, 1))


# In[14]:

third_fridays


# In[15]:

tf = lambda x: third_fridays[x]
vstoxx_futures['MATURITY'] = vstoxx_futures['EXP_MONTH'].apply(tf)


# In[16]:

vstoxx_futures.tail()


# ### VSTOXX Options Data

# In[17]:

vstoxx_options = pd.read_excel('./source/vstoxx_march_2014.xlsx',
                               'vstoxx_options')


# In[18]:

vstoxx_options.info()


# In[19]:

del vstoxx_options['A_SETTLEMENT_PRICE_SCALED']
del vstoxx_options['A_PRODUCT_ID']


# In[20]:

columns = ['DATE', 'EXP_YEAR', 'EXP_MONTH', 'TYPE', 'STRIKE', 'PRICE']
vstoxx_options.columns = columns


# In[21]:

vstoxx_options['MATURITY'] = vstoxx_options['EXP_MONTH'].apply(tf)


# In[22]:

vstoxx_options.head()


# In[23]:

vstoxx_options['STRIKE'] = vstoxx_options['STRIKE'] / 100.


# In[24]:

save = False
if save is True:
    import warnings
    warnings.simplefilter('ignore')
    h5 = pd.HDFStore('./source/vstoxx_march_2014.h5',
                     complevel=9, complib='blosc')
    h5['vstoxx_index'] = vstoxx_index
    h5['vstoxx_futures'] = vstoxx_futures
    h5['vstoxx_options'] = vstoxx_options
    h5.close()


# ## Model Calibration

# ### Relevant Market Data

# In[25]:

pricing_date = dt.datetime(2014, 3, 31)
  # last trading day in March 2014
maturity = third_fridays[10]
  # October maturity
initial_value = vstoxx_index['V2TX'][pricing_date]
  # VSTOXX on pricing_date
forward = vstoxx_futures[(vstoxx_futures.DATE == pricing_date)
            & (vstoxx_futures.MATURITY == maturity)]['PRICE'].values[0]


# In[26]:

tol = 0.20
option_selection =     vstoxx_options[(vstoxx_options.DATE == pricing_date)
                 & (vstoxx_options.MATURITY == maturity)
                 & (vstoxx_options.TYPE == 'C')
                 & (vstoxx_options.STRIKE > (1 - tol) * forward)
                 & (vstoxx_options.STRIKE < (1 + tol) * forward)]                            


# In[27]:

option_selection


# ### Option Modeling

# In[28]:

from dxa import *


# In[29]:

me_vstoxx = market_environment('me_vstoxx', pricing_date)


# In[30]:

me_vstoxx.add_constant('initial_value', initial_value)
me_vstoxx.add_constant('final_date', maturity)
me_vstoxx.add_constant('currency', 'EUR')


# In[31]:

me_vstoxx.add_constant('frequency', 'B')
me_vstoxx.add_constant('paths', 10000)


# In[32]:

csr = constant_short_rate('csr', 0.01)
  # somewhat arbitrarily chosen here


# In[33]:

me_vstoxx.add_curve('discount_curve', csr)


# In[34]:

# parameters to be calibrated later
me_vstoxx.add_constant('kappa', 1.0)
me_vstoxx.add_constant('theta', 1.2 * initial_value)
vol_est =  vstoxx_index['V2TX'].std()             * np.sqrt(len(vstoxx_index['V2TX']) / 252.)
me_vstoxx.add_constant('volatility', vol_est)


# In[35]:

vol_est


# In[36]:

vstoxx_model = square_root_diffusion('vstoxx_model', me_vstoxx)


# In[37]:

me_vstoxx.add_constant('strike', forward)
me_vstoxx.add_constant('maturity', maturity)


# In[38]:

payoff_func = 'np.maximum(maturity_value - strike, 0)'


# In[39]:

vstoxx_eur_call = valuation_mcs_european('vstoxx_eur_call',
                        vstoxx_model, me_vstoxx, payoff_func)


# In[40]:

vstoxx_eur_call.present_value()


# In[41]:

option_models = {}
for option in option_selection.index:
    strike = option_selection['STRIKE'].ix[option]
    me_vstoxx.add_constant('strike', strike)
    option_models[option] =                         valuation_mcs_european(
                                'eur_call_%d' % strike,
                                vstoxx_model,
                                me_vstoxx,
                                payoff_func)


# In[42]:

def calculate_model_values(p0):
    ''' Returns all relevant option values.
    
    Parameters
    ===========
    p0 : tuple/list
        tuple of kappa, theta, volatility
    
    Returns
    =======
    model_values : dict
        dictionary with model values
    '''
    kappa, theta, volatility = p0
    vstoxx_model.update(kappa=kappa,
                        theta=theta,
                        volatility=volatility)
    model_values = {}
    for option in option_models:
       model_values[option] =          option_models[option].present_value(fixed_seed=True)
    return model_values


# In[43]:

calculate_model_values((0.5, 27.5, vol_est))


# ### Calibration Procedure

# In[44]:

i = 0
def mean_squared_error(p0):
    ''' Returns the mean-squared error given
    the model and market values.
    
    Parameters
    ===========
    p0 : tuple/list
        tuple of kappa, theta, volatility
    
    Returns
    =======
    MSE : float
        mean-squared error
    '''
    global i
    model_values = np.array(calculate_model_values(p0).values())
    market_values = option_selection['PRICE'].values
    option_diffs = model_values - market_values
    MSE = np.sum(option_diffs ** 2) / len(option_diffs)
      # vectorized MSE calculation
    if i % 20 == 0:
        if i == 0:
            print '%4s  %6s  %6s  %6s --> %6s' %                  ('i', 'kappa', 'theta', 'vola', 'MSE')
        print '%4d  %6.3f  %6.3f  %6.3f --> %6.3f' %                 (i, p0[0], p0[1], p0[2], MSE)
    i += 1
    return MSE        


# In[45]:

mean_squared_error((0.5, 27.5, vol_est))


# In[46]:

import scipy.optimize as spo


# In[47]:

get_ipython().run_cell_magic('time', '', 'i = 0\nopt_global = spo.brute(mean_squared_error,\n                ((0.5, 3.01, 0.5),  # range for kappa\n                 (15., 30.1, 5.),  # range for theta\n                 (0.5, 5.51, 1)),  # range for volatility\n                 finish=None)')


# In[48]:

i = 0
mean_squared_error(opt_global)


# In[49]:

get_ipython().run_cell_magic('time', '', 'i = 0\nopt_local = spo.fmin(mean_squared_error, opt_global,\n                     xtol=0.00001, ftol=0.00001,\n                     maxiter=100, maxfun=350)')


# In[50]:

i = 0
mean_squared_error(opt_local)


# In[51]:

calculate_model_values(opt_local)


# In[52]:

pd.options.mode.chained_assignment = None
option_selection['MODEL'] =         np.array(calculate_model_values(opt_local).values())
option_selection['ERRORS'] =         option_selection['MODEL'] - option_selection['PRICE']


# In[53]:

option_selection[['MODEL', 'PRICE', 'ERRORS']]


# In[54]:

round(option_selection['ERRORS'].mean(), 3)


# In[55]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
fix, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 8))
strikes = option_selection['STRIKE'].values
ax1.plot(strikes, option_selection['PRICE'], label='market quotes')
ax1.plot(strikes, option_selection['MODEL'], 'ro', label='model values')
ax1.set_ylabel('option values')
ax1.grid(True)
ax1.legend(loc=0)
wi = 0.25
ax2.bar(strikes - wi / 2., option_selection['ERRORS'],
        label='market quotes', width=wi)
ax2.grid(True)
ax2.set_ylabel('differences')
ax2.set_xlabel('strikes')
# tag: vstoxx_calibration
# title: Calibrated model values for VSTOXX call options vs. market quotes


# ## American Options on the VSTOXX

# ### Modeling Option Positions

# In[56]:

me_vstoxx = market_environment('me_vstoxx', pricing_date)
me_vstoxx.add_constant('initial_value', initial_value)
me_vstoxx.add_constant('final_date', pricing_date)
me_vstoxx.add_constant('currency', 'NONE')


# In[57]:

# adding optimal parameters to environment
me_vstoxx.add_constant('kappa', opt_local[0])
me_vstoxx.add_constant('theta', opt_local[1])
me_vstoxx.add_constant('volatility', opt_local[2])


# In[58]:

me_vstoxx.add_constant('model', 'srd')


# In[59]:

payoff_func = 'np.maximum(strike - instrument_values, 0)'


# In[60]:

shared = market_environment('share', pricing_date)
shared.add_constant('maturity', maturity)
shared.add_constant('currency', 'EUR')


# In[61]:

option_positions = {}
  # dictionary for option positions
option_environments = {}
  # dictionary for option environments
for option in option_selection.index:
    option_environments[option] =         market_environment('am_put_%d' % option, pricing_date)
        # define new option environment, one for each option
    strike = option_selection['STRIKE'].ix[option]
      # pick the relevant strike
    option_environments[option].add_constant('strike', strike)
      # add it to the environment
    option_environments[option].add_environment(shared)
      # add the shared data
    option_positions['am_put_%d' % strike] =                     derivatives_position(
                        'am_put_%d' % strike,
                        quantity=100.,
                        underlying='vstoxx_model',
                        mar_env=option_environments[option],
                        otype='American',
                        payoff_func=payoff_func)


# ### The Options Portfolio

# In[62]:

val_env = market_environment('val_env', pricing_date)
val_env.add_constant('starting_date', pricing_date)
val_env.add_constant('final_date', pricing_date)
  # temporary value, is updated during valuation
val_env.add_curve('discount_curve', csr)
val_env.add_constant('frequency', 'B')
val_env.add_constant('paths', 25000)


# In[63]:

underlyings = {'vstoxx_model' : me_vstoxx}


# In[64]:

portfolio = derivatives_portfolio('portfolio', option_positions,
                                  val_env, underlyings)


# In[65]:

get_ipython().magic('time results = portfolio.get_statistics(fixed_seed=True)')


# In[66]:

results.sort(columns='name')


# In[67]:

results[['pos_value','pos_delta','pos_vega']].sum()


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
