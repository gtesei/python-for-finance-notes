
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


### Value of the European Call Option

S0 = 100.
K = 105.
T = 1.0
r = 0.05
sigma = 0.2

import numpy as np

I = 100000

np.random.seed(1000)
z = np.random.standard_normal(I)
ST = S0 * np.exp(( r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
hT = np.maximum(ST - K, 0)
C0 = np.exp(-r * T) * np.sum( hT) / I

print "Value of the European Call Option %5.3f" % C0


### Time-to-Results


import numpy as np
import pandas as pd
import pandas.io.data as web

goog = web.DataReader('GOOG', data_source='google',
                      start='3/14/2009', end='4/14/2014')
goog.index.name = u'Date'
print(goog.tail())


goog['Log_Ret'] = np.log(goog['Close'] / goog['Close'].shift(1))
goog['Volatility'] = pd.rolling_std(goog['Log_Ret'], window=252) * np.sqrt(252)


# plot
##get_ipython().magic('matplotlib inline')
goog[['Close', 'Volatility']].plot(subplots=True, color='blue',
                                   figsize=(8, 6), grid=True);
# tag: goog_vola
# title: Google closing prices and yearly volatility


### Paradigm

loops = 25000000
from math import *
a = range(1, loops)
def f(x):
    return 3 * log(x) + cos(x) ** 2

def npf(x):
    return 3 * np.log(x) + np.cos(x) ** 2


## std
import numpy as np
import time
tm = time.time()
a = np.arange(1, loops)
r = [f(x) for x in a]
pm = time.time() - tm
print 'time::',pm

## numpy
import numpy as np
import time
tm = time.time()
a = np.arange(1, loops)
r = [npf(x) for x in a]
pm = time.time() - tm
print 'time::',pm

##>>>>>>>>>>>>> numexpr
import numexpr as ne
ne.set_num_threads(4)
tm = time.time()
a = np.arange(1, loops)
r = ne.evaluate('3 * log(a) + cos(a) ** 2')
pm = time.time() - tm
print 'time::',pm
