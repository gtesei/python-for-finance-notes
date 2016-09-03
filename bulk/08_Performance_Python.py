
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

# # Performance Python

# **REMARK**: This notebook has been changed to Python **3.4**; needs a cluster to be running under 'default' profile for the IPython.parallel example.

# In[1]:

def perf_comp_data(func_list, data_list, rep=3, number=1):
    ''' Function to compare the performance of different functions.
    
    Parameters
    ==========
    func_list : list
        list with function names as strings
    data_list : list
        list with data set names as strings
    rep : int
        number of repetitions of the whole comparison
    number : int
        number of executions for every function
    '''
    from timeit import repeat
    res_list = {}
    for name in enumerate(func_list):
        stmt = name[1] + '(' + data_list[name[0]] + ')'
        setup = "from __main__ import " + name[1] + ', '                                     + data_list[name[0]]
        results = repeat(stmt=stmt, setup=setup,
                         repeat=rep, number=number)
        res_list[name[1]] = sum(results) / rep
    res_sort = sorted(res_list.items(),
                      key=lambda x: (x[1], x[0]))
    for item in res_sort:
        rel = item[1] / res_sort[0][1]
        print ('function: ' + item[0] +
              ', av. time sec: %9.5f, ' % item[1]
            + 'relative: %6.1f' % rel)


# ## Python Paradigms and Performance

# In[2]:

from math import *
def f(x):
    return abs(cos(x)) ** 0.5 + sin(2 + 3 * x)


# In[3]:

I = 500000
a_py = range(I)


# In[4]:

def f1(a):
    res = []
    for x in a:
        res.append(f(x))
    return res


# In[5]:

def f2(a):
    return [f(x) for x in a]


# In[6]:

def f3(a):
    ex = 'abs(cos(x)) ** 0.5 + sin(2 + 3 * x)'
    return [eval(ex) for x in a]


# In[7]:

import numpy as np


# In[8]:

a_np = np.arange(I)


# In[9]:

def f4(a):
    return (np.abs(np.cos(a)) ** 0.5 +
            np.sin(2 + 3 * a))


# In[10]:

import numexpr as ne


# In[11]:

def f5(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(1)
    return ne.evaluate(ex)


# In[12]:

def f6(a):
    ex = 'abs(cos(a)) ** 0.5 + sin(2 + 3 * a)'
    ne.set_num_threads(16)
    return ne.evaluate(ex)


# In[13]:

get_ipython().run_cell_magic('time', '', 'r1 = f1(a_py)\nr2 = f2(a_py)\nr3 = f3(a_py)\nr4 = f4(a_np)\nr5 = f5(a_np)\nr6 = f6(a_np)')


# In[14]:

np.allclose(r1, r2)


# In[15]:

np.allclose(r1, r3)


# In[16]:

np.allclose(r1, r4)


# In[17]:

np.allclose(r1, r5)


# In[18]:

np.allclose(r1, r6)


# In[19]:

func_list = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
data_list = ['a_py', 'a_py', 'a_py', 'a_np', 'a_np', 'a_np']


# In[20]:

perf_comp_data(func_list, data_list)


# ## Memory Layout and Performance

# In[21]:

import numpy as np


# In[22]:

np.zeros((3, 3), dtype=np.float64, order='C')


# In[23]:

c = np.array([[ 1.,  1.,  1.],
              [ 2.,  2.,  2.],
              [ 3.,  3.,  3.]], order='C')


# In[24]:

f = np.array([[ 1.,  1.,  1.],
              [ 2.,  2.,  2.],
              [ 3.,  3.,  3.]], order='F')


# In[25]:

x = np.random.standard_normal((3, 150000))
C = np.array(x, order='C')
F = np.array(x, order='F')
x = 0.0


# In[26]:

get_ipython().magic('timeit C.sum(axis=0)')


# In[27]:

get_ipython().magic('timeit C.sum(axis=1)')


# In[28]:

get_ipython().magic('timeit C.std(axis=0)')


# In[29]:

get_ipython().magic('timeit C.std(axis=1)')


# In[30]:

get_ipython().magic('timeit F.sum(axis=0)')


# In[31]:

get_ipython().magic('timeit F.sum(axis=1)')


# In[32]:

get_ipython().magic('timeit F.std(axis=0)')


# In[33]:

get_ipython().magic('timeit F.std(axis=1)')


# In[34]:

C = 0.0; F = 0.0


# ## Parallel Computing

# ### The Monte Carlo Algorithm

# In[35]:

def bsm_mcs_valuation(strike):
    ''' Dynamic Black-Scholes-Merton Monte Carlo estimator
    for European calls.
    
    Parameters
    ==========
    strike : float
        strike price of the option
    
    Results
    =======
    value : float
        estimate for present value of call option
    '''
    import numpy as np
    S0 = 100.; T = 1.0; r = 0.05; vola = 0.2
    M = 50; I = 20000
    dt = T / M
    rand = np.random.standard_normal((M + 1, I))
    S = np.zeros((M + 1, I)); S[0] = S0
    for t in range(1, M + 1):
        S[t] = S[t-1] * np.exp((r - 0.5 * vola ** 2) * dt
                               + vola * np.sqrt(dt) * rand[t])
    value = (np.exp(-r * T)
                     * np.sum(np.maximum(S[-1] - strike, 0)) / I)
    return value


# ### The Sequential Calculation

# In[36]:

def seq_value(n):
    ''' Sequential option valuation.
    
    Parameters
    ==========
    n : int
        number of option valuations/strikes
    '''
    strikes = np.linspace(80, 120, n)
    option_values = []
    for strike in strikes:
        option_values.append(bsm_mcs_valuation(strike))
    return strikes, option_values


# In[37]:

n = 100  # number of options to be valued
get_ipython().magic('time strikes, option_values_seq = seq_value(n)')


# In[38]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.figure(figsize=(8, 4))
plt.plot(strikes, option_values_seq, 'b')
plt.plot(strikes, option_values_seq, 'r.')
plt.grid(True)
plt.xlabel('strikes')
plt.ylabel('European call option values')
# tag: option_values
# title: European call option values by Monte Carlo simulation
# size: 60


# ### The Parallel Calculation

# In[39]:

from IPython.parallel import Client
c = Client(profile="default")
view = c.load_balanced_view()


# In[40]:

def par_value(n):
    ''' Sequential option valuation.
    
    Parameters
    ==========
    n : int
        number of option valuations/strikes
    '''
    strikes = np.linspace(80, 120, n)
    option_values = []
    for strike in strikes:
        value = view.apply_async(bsm_mcs_valuation, strike)
        option_values.append(value)
    c.wait(option_values)
    return strikes, option_values


# In[41]:

get_ipython().magic('time strikes, option_values_obj = par_value(n)')


# In[42]:

option_values_obj[0].metadata


# In[43]:

option_values_obj[0].result


# In[44]:

option_values_par = []
for res in option_values_obj:
    option_values_par.append(res.result)


# In[45]:

plt.figure(figsize=(8, 4))
plt.plot(strikes, option_values_seq, 'b', label='Sequential')
plt.plot(strikes, option_values_par, 'r.', label='Parallel')
plt.grid(True); plt.legend(loc=0)
plt.xlabel('strikes')
plt.ylabel('European call option values')
# tag: option_comp
# title: Comparison of European call option values
# size: 60


# ### Performance Comparison

# In[46]:

n = 50  # number of option valuations
func_list = ['seq_value', 'par_value']
data_list = 2 * ['n']


# In[47]:

perf_comp_data(func_list, data_list)


# ## Multiprocessing

# In[48]:

import multiprocessing as mp


# In[49]:

import math
def simulate_geometric_brownian_motion(p):
    M, I = p
      # time steps, paths
    S0 = 100; r = 0.05; sigma = 0.2; T = 1.0
      # model parameters
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                    sigma * math.sqrt(dt) * np.random.standard_normal(I))
    return paths


# In[50]:

paths = simulate_geometric_brownian_motion((5, 2))
paths


# In[51]:

I = 10000  # number of paths
M = 50  # number of time steps
t = 20  # number of tasks/simulations


# In[52]:

# running on server with 4 cores
from time import time
times = []
for w in range(1, 5):
    t0 = time()
    pool = mp.Pool(processes=w)
      # the pool of workers
    result = pool.map(simulate_geometric_brownian_motion, t * [(M, I), ])
      # the mapping of the function to the list of parameter tuples
    times.append(time() - t0)


# In[53]:

plt.plot(range(1, 5), times)
plt.plot(range(1, 5), times, 'ro')
plt.grid(True)
plt.xlabel('number of processes')
plt.ylabel('time in seconds')
plt.title('%d Monte Carlo simulations' % t)
# tag: multi_proc
# title: Comparison execution speed dependent on the number of threads used (4 core machine)
# size: 60


# ## Dynamic Compiling

# ### Introductory Example

# In[54]:

from math import cos, log
def f_py(I, J):
    res = 0
    for i in range(I):
        for j in range (J):
            res += int(cos(log(1)))
    return res


# In[55]:

I, J = 2500, 2500
get_ipython().magic('time f_py(I, J)')


# In[56]:

def f_np(I, J):
    a = np.ones((I, J), dtype=np.float64)
    return int(np.sum(np.cos(np.log(a)))), a


# In[57]:

get_ipython().magic('time res, a = f_np(I, J)')


# In[58]:

a.nbytes


# In[59]:

import numba as nb


# In[60]:

f_nb = nb.jit(f_py)


# In[61]:

get_ipython().magic('time f_nb(I, J)')


# In[62]:

func_list = ['f_py', 'f_np', 'f_nb']
data_list = 3 * ['I, J']


# In[63]:

perf_comp_data(func_list, data_list)


# ### Binomial Option Pricing

# In[64]:

# model & option Parameters
S0 = 100.  # initial index level
T = 1.  # call option maturity
r = 0.05  # constant short rate
vola = 0.20  # constant volatility factor of diffusion

# time parameters
M = 1000  # time steps
dt = T / M  # length of time interval
df = exp(-r * dt)  # discount factor per time interval

# binomial parameters
u = exp(vola * sqrt(dt))  # up-movement
d = 1 / u  # down-movement
q = (exp(r * dt) - d) / (u - d)  # martingale probability


# In[65]:

import numpy as np
def binomial_py(strike):
    ''' Binomial option pricing via looping.
    
    Parameters
    ==========
    strike : float
        strike price of the European call option
    '''
    # LOOP 1 - Index Levels
    S = np.zeros((M + 1, M + 1), dtype=np.float64)
      # index level array
    S[0, 0] = S0
    z1 = 0
    for j in range(1, M + 1, 1):
        z1 = z1 + 1
        for i in range(z1 + 1):
            S[i, j] = S[0, 0] * (u ** j) * (d ** (i * 2))
            
    # LOOP 2 - Inner Values
    iv = np.zeros((M + 1, M + 1), dtype=np.float64)
      # inner value array
    z2 = 0
    for j in range(0, M + 1, 1):
        for i in range(z2 + 1):
            iv[i, j] = max(S[i, j] - strike, 0)
        z2 = z2 + 1
        
    # LOOP 3 - Valuation
    pv = np.zeros((M + 1, M + 1), dtype=np.float64)
      # present value array
    pv[:, M] = iv[:, M]  # initialize last time point
    z3 = M + 1
    for j in range(M - 1, -1, -1):
        z3 = z3 - 1
        for i in range(z3):
            pv[i, j] = (q * pv[i, j + 1] +
                        (1 - q) * pv[i + 1, j + 1]) * df
    return pv[0, 0]


# In[66]:

get_ipython().magic('time round(binomial_py(100), 3)')


# In[67]:

get_ipython().magic('time round(bsm_mcs_valuation(100), 3)')


# In[68]:

def binomial_np(strike):
    ''' Binomial option pricing with NumPy.
    
    Parameters
    ==========
    strike : float
        strike price of the European call option
    '''
    # Index Levels with NumPy
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md
    
    # Valuation Loop
    pv = np.maximum(S - strike, 0)

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        pv[0:M - z, t] = (q * pv[0:M - z, t + 1]
                        + (1 - q) * pv[1:M - z + 1, t + 1]) * df
        z += 1
    return pv[0, 0]


# In[69]:

M = 4  # four time steps only
mu = np.arange(M + 1)
mu


# In[70]:

mu = np.resize(mu, (M + 1, M + 1))
mu


# In[71]:

md = np.transpose(mu)
md


# In[72]:

mu = u ** (mu - md)
mu.round(3)


# In[73]:

md = d ** md
md.round(3)


# In[74]:

S = S0 * mu * md
S.round(3)


# In[75]:

M = 1000  # reset number of time steps
get_ipython().magic('time round(binomial_np(100), 3)')


# In[76]:

binomial_nb = nb.jit(binomial_py)


# In[77]:

get_ipython().magic('time round(binomial_nb(100), 3)')


# In[78]:

func_list = ['binomial_py', 'binomial_np', 'binomial_nb']
K = 100.
data_list = 3 * ['K']


# In[79]:

perf_comp_data(func_list, data_list)


# ## Static Compiling with Cython

# In[80]:

def f_py(I, J):
    res = 0.  # we work on a float object
    for i in range(I):
        for j in range (J * I):
            res += 1
    return res


# In[81]:

I, J = 500, 500
get_ipython().magic('time f_py(I, J)')


# In[82]:

import pyximport
pyximport.install()


# In[83]:

import sys
sys.path.append('data/')
  # path to the Cython script
  # not needed if in same directory


# In[84]:

from nested_loop import f_cy


# In[85]:

get_ipython().magic('time res = f_cy(I, J)')


# In[86]:

res


# In[87]:

get_ipython().magic('load_ext Cython')


# In[88]:

get_ipython().run_cell_magic('cython', '', '#\n# Nested loop example with Cython\n#\ndef f_cy(int I, int J):\n    cdef double res = 0\n    # double float much slower than int or long\n    for i in range(I):\n        for j in range (J * I):\n            res += 1\n    return res')


# In[89]:

get_ipython().magic('time res = f_cy(I, J)')


# In[90]:

res


# In[91]:

import numba as nb


# In[92]:

f_nb = nb.jit(f_py)


# In[93]:

get_ipython().magic('time res = f_nb(I, J)')


# In[94]:

res


# In[95]:

func_list = ['f_py', 'f_cy', 'f_nb']
I, J = 500, 500
data_list = 3 * ['I, J']


# In[96]:

perf_comp_data(func_list, data_list)


# ## Generation of Random Numbers on GPUs

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
