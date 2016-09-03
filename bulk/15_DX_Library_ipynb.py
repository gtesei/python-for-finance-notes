
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

# # DX Library

# In[1]:

import numpy as np
import pandas as pd
import datetime as dt


# In[2]:

import sys
sys.path.append('../python/dxa')


# ## DX Frame

# In[3]:

from dx_frame import *


# ### Risk-Neutral Discounting

# In[4]:

dates = [dt.datetime(2015, 1, 1), dt.datetime(2015, 7, 1), dt.datetime(2016, 1, 1)]


# In[5]:

deltas = [0.0, 0.5, 1.0]


# In[6]:

csr = constant_short_rate('csr', 0.05)


# In[7]:

csr.get_discount_factors(dates)


# In[8]:

deltas = get_year_deltas(dates)
deltas


# In[9]:

csr.get_discount_factors(deltas, dtobjects=False)


# ### Market Environment

# In[10]:

me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))


# In[11]:

me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('final_date', dt.datetime(2015, 12, 31))
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('frequency', 'M')
me_gbm.add_constant('paths', 10000)


# In[12]:

me_gbm.add_curve('discount_curve', csr)


# ## DX Simulation

# In[13]:

from sn_random_numbers import *


# In[14]:

snrn = sn_random_numbers((2, 2, 2), antithetic=False, moment_matching=False, fixed_seed=True)
snrn


# In[15]:

snrn = sn_random_numbers((2, 2, 2), antithetic=False, moment_matching=True, fixed_seed=True)
snrn


# In[16]:

snrn.mean()


# ### Geometric Brownian Motion

# In[17]:

from dx_frame import *


# In[18]:

me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))


# In[19]:

me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('final_date', dt.datetime(2015, 12, 31))
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('frequency', 'M')
  # monthly frequency (respcective month end)
me_gbm.add_constant('paths', 10000)


# In[20]:

csr = constant_short_rate('csr', 0.06)


# In[21]:

me_gbm.add_curve('discount_curve', csr)


# In[22]:

from geometric_brownian_motion import geometric_brownian_motion


# In[23]:

gbm = geometric_brownian_motion('gbm', me_gbm)


# In[24]:

gbm.generate_time_grid()


# In[25]:

gbm.time_grid


# In[26]:

get_ipython().magic('time paths_1 = gbm.get_instrument_values()')


# In[27]:

paths_1


# In[28]:

gbm.update(volatility=0.5)


# In[29]:

get_ipython().magic('time paths_2 = gbm.get_instrument_values()')


# In[30]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.figure(figsize=(8, 4))
p1 = plt.plot(gbm.time_grid, paths_1[:, :10], 'b')
p2 = plt.plot(gbm.time_grid, paths_2[:, :10], 'r-.')
plt.grid(True)
l1 = plt.legend([p1[0], p2[0]],
                ['low volatility', 'high volatility'], loc=2)
plt.gca().add_artist(l1)
plt.xticks(rotation=30)
# tag: gbm_class_fig
# title: Simulated paths from geometric Brownian motion simulation class


# ### Jump Diffusion

# In[31]:

me_jd = market_environment('me_jd', dt.datetime(2015, 1, 1))


# In[32]:

# specific to simulation class
me_jd.add_constant('lambda', 0.3)
me_jd.add_constant('mu', -0.75)
me_jd.add_constant('delta', 0.1)


# In[33]:

me_jd.add_environment(me_gbm)


# In[34]:

from jump_diffusion import jump_diffusion


# In[35]:

jd = jump_diffusion('jd', me_jd)


# In[36]:

get_ipython().magic('time paths_3 = jd.get_instrument_values()')


# In[37]:

jd.update(lamb=0.9)


# In[38]:

get_ipython().magic('time paths_4 = jd.get_instrument_values()')


# In[39]:

plt.figure(figsize=(8, 4))
p1 = plt.plot(gbm.time_grid, paths_3[:, :10], 'b')
p2 = plt.plot(gbm.time_grid, paths_4[:, :10], 'r-.')
plt.grid(True)
l1 = plt.legend([p1[0], p2[0]],
                ['low intensity', 'high intensity'], loc=3)
plt.gca().add_artist(l1)
plt.xticks(rotation=30)
# tag: jd_class_fig
# title: Simulated paths from jump diffusion simulation class


# ### Square-Root Diffusion

# In[40]:

me_srd = market_environment('me_srd', dt.datetime(2015, 1, 1))


# In[41]:

me_srd.add_constant('initial_value', .25)
me_srd.add_constant('volatility', 0.05)
me_srd.add_constant('final_date', dt.datetime(2015, 12, 31))
me_srd.add_constant('currency', 'EUR')
me_srd.add_constant('frequency', 'W')
me_srd.add_constant('paths', 10000)


# In[42]:

# specific to simualation class
me_srd.add_constant('kappa', 4.0)
me_srd.add_constant('theta', 0.2)


# In[43]:

# required but not needed for the class
me_srd.add_curve('discount_curve', constant_short_rate('r', 0.0))


# In[44]:

from square_root_diffusion import square_root_diffusion


# In[45]:

srd = square_root_diffusion('srd', me_srd)


# In[46]:

srd_paths = srd.get_instrument_values()[:, :10]


# In[47]:

plt.figure(figsize=(8, 4))
plt.plot(srd.time_grid, srd.get_instrument_values()[:, :10])
plt.axhline(me_srd.get_constant('theta'), color='r', ls='--', lw=2.0)
plt.grid(True)
plt.xticks(rotation=30)
# tag: srd_class_fig
# title: Simulated paths from square-root diffusion simulation class (dashed line = long-term mean)


# ## Valuation Classes

# ### European Options

# In[48]:

from dx_simulation import *


# In[49]:

me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))


# In[50]:

me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('final_date', dt.datetime(2015, 12, 31))
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('frequency', 'M')
me_gbm.add_constant('paths', 10000)


# In[51]:

csr = constant_short_rate('csr', 0.06)


# In[52]:

me_gbm.add_curve('discount_curve', csr)


# In[53]:

gbm = geometric_brownian_motion('gbm', me_gbm)


# In[54]:

me_call = market_environment('me_call', me_gbm.pricing_date)


# In[55]:

me_call.add_constant('strike', 40.)
me_call.add_constant('maturity', dt.datetime(2015, 12, 31))
me_call.add_constant('currency', 'EUR')


# In[56]:

payoff_func = 'np.maximum(maturity_value - strike, 0)'


# In[57]:

from valuation_mcs_european import valuation_mcs_european


# In[58]:

eur_call = valuation_mcs_european('eur_call', underlying=gbm,
                        mar_env=me_call, payoff_func=payoff_func)


# In[59]:

get_ipython().magic('time eur_call.present_value()')


# In[60]:

get_ipython().magic('time eur_call.delta()')


# In[61]:

get_ipython().magic('time eur_call.vega()')


# In[62]:

get_ipython().run_cell_magic('time', '', 's_list = np.arange(34., 46.1, 2.)\np_list = []; d_list = []; v_list = []\nfor s in s_list:\n    eur_call.update(initial_value=s)\n    p_list.append(eur_call.present_value(fixed_seed=True))\n    d_list.append(eur_call.delta())\n    v_list.append(eur_call.vega())')


# In[63]:

from plot_option_stats import plot_option_stats
get_ipython().magic('matplotlib inline')


# In[64]:

plot_option_stats(s_list, p_list, d_list, v_list)
# tag: option_stats_1
# title: Present value, Delta and Vega estimates for European call option
# size: 75


# In[65]:

payoff_func = 'np.maximum(0.33 * (maturity_value + max_value) - 40, 0)'
  # payoff dependent on both the simulated maturity value
  # and the maximum value


# In[66]:

eur_as_call = valuation_mcs_european('eur_as_call', underlying=gbm,
                            mar_env=me_call, payoff_func=payoff_func)


# In[67]:

get_ipython().run_cell_magic('time', '', 's_list = np.arange(34., 46.1, 2.)\np_list = []; d_list = []; v_list = []\nfor s in s_list:\n    eur_as_call.update(s)\n    p_list.append(eur_as_call.present_value(fixed_seed=True))\n    d_list.append(eur_as_call.delta())\n    v_list.append(eur_as_call.vega())')


# In[68]:

plot_option_stats(s_list, p_list, d_list, v_list)
# tag: option_stats_2
# title: Present value, Delta and Vega estimates for European Asian call option
# size: 75


# ### American Options

# In[69]:

from dx_simulation import *


# In[70]:

me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))


# In[71]:

me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('final_date', dt.datetime(2016, 12, 31))
me_gbm.add_constant('currency', 'EUR')
me_gbm.add_constant('frequency', 'W')
  # weekly frequency
me_gbm.add_constant('paths', 50000)


# In[72]:

csr = constant_short_rate('csr', 0.06)


# In[73]:

me_gbm.add_curve('discount_curve', csr)


# In[74]:

gbm = geometric_brownian_motion('gbm', me_gbm)


# In[75]:

payoff_func = 'np.maximum(strike - instrument_values, 0)'


# In[76]:

me_am_put = market_environment('me_am_put', dt.datetime(2015, 1, 1))


# In[77]:

me_am_put.add_constant('maturity', dt.datetime(2015, 12, 31))
me_am_put.add_constant('strike', 40.)
me_am_put.add_constant('currency', 'EUR')


# In[78]:

from valuation_mcs_american import valuation_mcs_american


# In[79]:

am_put = valuation_mcs_american('am_put', underlying=gbm,
                    mar_env=me_am_put, payoff_func=payoff_func)


# In[80]:

get_ipython().magic('time am_put.present_value(fixed_seed=True, bf=5)')


# In[81]:

get_ipython().run_cell_magic('time', '', 'ls_table = []\nfor initial_value in (36., 38., 40., 42., 44.): \n    for volatility in (0.2, 0.4):\n        for maturity in (dt.datetime(2015, 12, 31),\n                         dt.datetime(2016, 12, 31)):\n            am_put.update(initial_value=initial_value,\n                          volatility=volatility,\n                          maturity=maturity)\n            ls_table.append([initial_value,\n                             volatility,\n                             maturity,\n                             am_put.present_value(bf=5)])')


# In[82]:

print "S0  | Vola | T | Value"
print 22 * "-"
for r in ls_table:
    print "%d  | %3.1f  | %d | %5.3f" %           (r[0], r[1], r[2].year - 2014, r[3])


# In[83]:

am_put.update(initial_value=36.)
am_put.delta()


# In[84]:

am_put.vega()


# ## Portfolios

# ### Position

# In[85]:

from dx_valuation import *


# In[86]:

me_gbm = market_environment('me_gbm', dt.datetime(2015, 1, 1))


# In[87]:

me_gbm.add_constant('initial_value', 36.)
me_gbm.add_constant('volatility', 0.2)
me_gbm.add_constant('currency', 'EUR')


# In[88]:

me_gbm.add_constant('model', 'gbm')


# In[89]:

from derivatives_position import derivatives_position


# In[90]:

me_am_put = market_environment('me_am_put', dt.datetime(2015, 1, 1))


# In[91]:

me_am_put.add_constant('maturity', dt.datetime(2015, 12, 31))
me_am_put.add_constant('strike', 40.)
me_am_put.add_constant('currency', 'EUR')


# In[92]:

payoff_func = 'np.maximum(strike - instrument_values, 0)'


# In[93]:

am_put_pos = derivatives_position(
             name='am_put_pos',
             quantity=3,
             underlying='gbm',
             mar_env=me_am_put,
             otype='American',
             payoff_func=payoff_func)


# In[94]:

am_put_pos.get_info()


# #### Portfolio

# In[95]:

me_jd = market_environment('me_jd', me_gbm.pricing_date)


# In[96]:

# add jump diffusion specific parameters
me_jd.add_constant('lambda', 0.3)
me_jd.add_constant('mu', -0.75)
me_jd.add_constant('delta', 0.1)
# add other parameters from gbm
me_jd.add_environment(me_gbm)


# In[97]:

# needed for portfolio valuation
me_jd.add_constant('model', 'jd')


# In[98]:

me_eur_call = market_environment('me_eur_call', me_jd.pricing_date)


# In[99]:

me_eur_call.add_constant('maturity', dt.datetime(2015, 6, 30))
me_eur_call.add_constant('strike', 38.)
me_eur_call.add_constant('currency', 'EUR')


# In[100]:

payoff_func = 'np.maximum(maturity_value - strike, 0)'


# In[101]:

eur_call_pos = derivatives_position(
             name='eur_call_pos',
             quantity=5,
             underlying='jd',
             mar_env=me_eur_call,
             otype='European',
             payoff_func=payoff_func)


# In[102]:

underlyings = {'gbm': me_gbm, 'jd' : me_jd}
positions = {'am_put_pos' : am_put_pos, 'eur_call_pos' : eur_call_pos}


# In[103]:

# discounting object for the valuation
csr = constant_short_rate('csr', 0.06)


# In[104]:

val_env = market_environment('general', me_gbm.pricing_date)
val_env.add_constant('frequency', 'W')
  # monthly frequency
val_env.add_constant('paths', 25000)
val_env.add_constant('starting_date', val_env.pricing_date)
val_env.add_constant('final_date', val_env.pricing_date)
  # not yet known; take pricing_date temporarily
val_env.add_curve('discount_curve', csr)
  # select single discount_curve for whole portfolio


# In[105]:

from derivatives_portfolio import derivatives_portfolio


# In[106]:

portfolio = derivatives_portfolio(
                name='portfolio',
                positions=positions,
                val_env=val_env,
                assets=underlyings,
                fixed_seed=False)


# In[107]:

portfolio.get_statistics(fixed_seed=False)


# In[108]:

portfolio.get_statistics(fixed_seed=False)[['pos_value', 'pos_delta', 'pos_vega']].sum()
  # aggregate over all positions


# In[109]:

# portfolio.get_positions()


# In[110]:

portfolio.valuation_objects['am_put_pos'].present_value()


# In[111]:

portfolio.valuation_objects['eur_call_pos'].delta()


# In[112]:

path_no = 777
path_gbm = portfolio.underlying_objects['gbm'].get_instrument_values()[
                                                            :, path_no]
path_jd = portfolio.underlying_objects['jd'].get_instrument_values()[
                                                            :, path_no]


# In[113]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[114]:

plt.figure(figsize=(7, 4))
plt.plot(portfolio.time_grid, path_gbm, 'r', label='gbm')
plt.plot(portfolio.time_grid, path_jd, 'b', label='jd')
plt.xticks(rotation=30)
plt.legend(loc=0); plt.grid(True)
# tag: dx_portfolio_1
# title: Non-correlated risk factors


# In[115]:

correlations = [['gbm', 'jd', 0.9]]


# In[116]:

port_corr = derivatives_portfolio(
                name='portfolio',
                positions=positions,
                val_env=val_env,
                assets=underlyings,
                correlations=correlations,
                fixed_seed=True)


# In[117]:

port_corr.get_statistics()


# In[118]:

path_gbm = port_corr.underlying_objects['gbm'].            get_instrument_values()[:, path_no]
path_jd = port_corr.underlying_objects['jd'].            get_instrument_values()[:, path_no]


# In[119]:

plt.figure(figsize=(7, 4))
plt.plot(portfolio.time_grid, path_gbm, 'r', label='gbm')
plt.plot(portfolio.time_grid, path_jd, 'b', label='jd')
plt.xticks(rotation=30)
plt.legend(loc=0); plt.grid(True)
# tag: dx_portfolio_2
# title: Highly correlated risk factors


# In[120]:

pv1 = 5 * port_corr.valuation_objects['eur_call_pos'].            present_value(full=True)[1]
pv1


# In[121]:

pv2 = 3 * port_corr.valuation_objects['am_put_pos'].            present_value(full=True)[1]
pv2


# In[122]:

plt.hist([pv1, pv2], bins=25,
         label=['European call', 'American put']);
plt.axvline(pv1.mean(), color='r', ls='dashed',
            lw=1.5, label='call mean = %4.2f' % pv1.mean())
plt.axvline(pv2.mean(), color='r', ls='dotted',
            lw=1.5, label='put mean = %4.2f' % pv2.mean())
plt.xlim(0, 80); plt.ylim(0, 10000)
plt.grid(); plt.legend()
# tag: dx_portfolio_3
# title: Frequency distributions of option position present values


# In[123]:

pvs = pv1 + pv2
plt.hist(pvs, bins=50, label='portfolio');
plt.axvline(pvs.mean(), color='r', ls='dashed',
            lw=1.5, label='mean = %4.2f' % pvs.mean())
plt.xlim(0, 80); plt.ylim(0, 7000)
plt.grid(); plt.legend()
# tag: dx_portfolio_4
# title: Portfolio frequency distribution of present values


# In[124]:

# portfolio with correlation
pvs.std()


# In[125]:

# portfolio without correlation
pv1 = 5 * portfolio.valuation_objects['eur_call_pos'].            present_value(full=True)[1]
pv2 = 3 * portfolio.valuation_objects['am_put_pos'].            present_value(full=True)[1]
(pv1 + pv2).std()


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
