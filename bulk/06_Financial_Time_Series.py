
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

# # Financial Time Series

# In[1]:

import warnings
warnings.simplefilter('ignore')


# ## pandas Basics

# In[2]:

import numpy as np
import pandas as pd


# ### First Steps with DataFrame Class

# In[3]:

df = pd.DataFrame([10, 20, 30, 40], columns=['numbers'],
                  index=['a', 'b', 'c', 'd'])
df


# In[4]:

df.index  # the index values


# In[5]:

df.columns  # the column names


# In[6]:

df.ix['c']  # selection via index


# In[7]:

df.ix[['a', 'd']]  # selection of multiple indices


# In[8]:

df.ix[df.index[1:3]]  # selection via Index object


# In[9]:

df.sum()  # sum per column


# In[10]:

df.apply(lambda x: x ** 2)  # square of every element


# In[11]:

df ** 2  # again square, this time NumPy-like


# In[12]:

df['floats'] = (1.5, 2.5, 3.5, 4.5)
  # new column is generated
df


# In[13]:

df['floats']  # selection of column


# In[14]:

df['names'] = pd.DataFrame(['Yves', 'Guido', 'Felix', 'Francesc'],
                           index=['d', 'a', 'b', 'c'])
df


# In[15]:

df.append({'numbers': 100, 'floats': 5.75, 'names': 'Henry'},
               ignore_index=True)
  # temporary object; df not changed


# In[16]:

df = df.append(pd.DataFrame({'numbers': 100, 'floats': 5.75,
                             'names': 'Henry'}, index=['z',]))
df


# In[17]:

df.join(pd.DataFrame([1, 4, 9, 16, 25],
            index=['a', 'b', 'c', 'd', 'y'],
            columns=['squares',]))
  # temporary object


# In[18]:

df = df.join(pd.DataFrame([1, 4, 9, 16, 25],
                    index=['a', 'b', 'c', 'd', 'y'],
                    columns=['squares',]),
                    how='outer')
df


# In[19]:

df[['numbers', 'squares']].mean()
  # column-wise mean


# In[20]:

df[['numbers', 'squares']].std()
  # column-wise standard deviation


# ### Second Steps with DataFrame Class

# In[21]:

a = np.random.standard_normal((9, 4))
a.round(6)


# In[22]:

df = pd.DataFrame(a)
df


# In[23]:

df.columns = [['No1', 'No2', 'No3', 'No4']]
df


# In[24]:

df['No2'][3]  # value in column No2 at index position 3


# In[25]:

dates = pd.date_range('2015-1-1', periods=9, freq='M')
dates


# In[26]:

df.index = dates
df


# In[27]:

np.array(df).round(6)


# ### Basic Analytics

# In[28]:

df.sum()


# In[29]:

df.mean()


# In[30]:

df.cumsum()


# In[31]:

df.describe()


# In[32]:

np.sqrt(df)


# In[33]:

np.sqrt(df).sum()


# In[34]:

get_ipython().magic('matplotlib inline')
df.cumsum().plot(lw=2.0, grid=True)
# tag: dataframe_plot
# title: Line plot of a DataFrame object


# ### Series Class

# In[35]:

type(df)


# In[36]:

df['No1']


# In[37]:

type(df['No1'])


# In[38]:

import matplotlib.pyplot as plt
df['No1'].cumsum().plot(style='r', lw=2., grid=True)
plt.xlabel('date')
plt.ylabel('value')
# tag: time_series
# title: Line plot of a Series object


# ### GroupBy Operations

# In[39]:

df['Quarter'] = ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3']
df


# In[40]:

groups = df.groupby('Quarter')


# In[41]:

groups.mean()


# In[42]:

groups.max()


# In[43]:

groups.size()


# In[44]:

df['Odd_Even'] = ['Odd', 'Even', 'Odd', 'Even', 'Odd', 'Even',
                  'Odd', 'Even', 'Odd']


# In[45]:

groups = df.groupby(['Quarter', 'Odd_Even'])


# In[46]:

groups.size()


# In[47]:

groups.mean()


# ## Financial Data

# In[48]:

import pandas.io.data as web


# In[49]:

DAX = web.DataReader(name='^GDAXI', data_source='yahoo',
                     start='2000-1-1')
DAX.info()


# In[50]:

DAX.tail()


# In[51]:

DAX['Close'].plot(figsize=(8, 5), grid=True)
# tag: dax
# title: Historical DAX index levels

%%time
DAX['Ret_Loop'] = 0.0
for i in range(1, len(DAX)):
    DAX['Ret_Loop'][i] = np.log(DAX['Close'][i] /
                                DAX['Close'][i - 1])DAX[['Close', 'Ret_Loop']].tail()
# In[52]:

get_ipython().magic("time DAX['Return'] = np.log(DAX['Close'] / DAX['Close'].shift(1))")


# In[53]:

DAX[['Close', 'Return', 'Return']].tail()

del DAX['Ret_Loop']
# In[54]:

DAX[['Close', 'Return']].plot(subplots=True, style='b',
                              figsize=(8, 5), grid=True)
# tag: dax_returns
# title: The DAX index and daily log returns


# In[55]:

DAX['42d'] = pd.rolling_mean(DAX['Close'], window=42)
DAX['252d'] = pd.rolling_mean(DAX['Close'], window=252)


# In[56]:

DAX[['Close', '42d', '252d']].tail()


# In[57]:

DAX[['Close', '42d', '252d']].plot(figsize=(8, 5), grid=True)
# tag: dax_trends
# title: The DAX index and moving averages


# In[58]:

import math
DAX['Mov_Vol'] = pd.rolling_std(DAX['Return'],
                                window=252) * math.sqrt(252)
  # moving annual volatility


# In[59]:

DAX[['Close', 'Mov_Vol', 'Return']].plot(subplots=True, style='b',
                                         figsize=(8, 7), grid=True)
# tag: dax_mov_std
# title: The DAX index and moving, annualized volatility


# ## Regression Analysis

# In[60]:

import pandas as pd
from urllib import urlretrieve


# In[61]:

es_url = 'https://www.stoxx.com/document/Indices/Current/HistoricalData/hbrbcpe.txt'
vs_url = 'https://www.stoxx.com/document/Indices/Current/HistoricalData/h_vstoxx.txt'
urlretrieve(es_url, './data/es.txt')
urlretrieve(vs_url, './data/vs.txt')
get_ipython().system('ls -o ./data/*.txt')
# Windows: use dir


# In[62]:

lines = open('./data/es.txt', 'r').readlines()
lines = [line.replace(' ', '') for line in lines]


# In[63]:

lines[:6]


# In[64]:

for line in lines[3883:3890]:
    print line[41:],


# In[65]:

new_file = open('./data/es50.txt', 'w')
    # opens a new file
new_file.writelines('date' + lines[3][:-1]
                    + ';DEL' + lines[3][-1])
    # writes the corrected third line of the orginal file
    # as first line of new file
new_file.writelines(lines[4:])
    # writes the remaining lines of the orginial file
new_file.close()


# In[66]:

new_lines = open('./data/es50.txt', 'r').readlines()
new_lines[:5]


# In[67]:

es = pd.read_csv('./data/es50.txt', index_col=0,
                 parse_dates=True, sep=';', dayfirst=True)


# In[68]:

np.round(es.tail())


# In[69]:

del es['DEL'] 
es.info()


# In[70]:

cols = ['SX5P', 'SX5E', 'SXXP', 'SXXE', 'SXXF',
        'SXXA', 'DK5F', 'DKXF']
es = pd.read_csv(es_url, index_col=0, parse_dates=True,
                 sep=';', dayfirst=True, header=None,
                 skiprows=4, names=cols)


# In[71]:

es.tail()


# In[72]:

vs = pd.read_csv('./data/vs.txt', index_col=0, header=2,
                 parse_dates=True, dayfirst=True)
vs.info()


# In[73]:

import datetime as dt
data = pd.DataFrame({'EUROSTOXX' :
                     es['SX5E'][es.index > dt.datetime(1999, 1, 1)]})
data = data.join(pd.DataFrame({'VSTOXX' :
                     vs['V2TX'][vs.index > dt.datetime(1999, 1, 1)]}))


# In[74]:

data = data.fillna(method='ffill')
data.info()


# In[75]:

data.tail()


# In[76]:

data.plot(subplots=True, grid=True, style='b', figsize=(8, 6))
# tag: es50_vs
# title: The EURO STOXX 50 Index and the VSTOXX volatility index


# In[77]:

rets = np.log(data / data.shift(1)) 
rets.head()


# In[78]:

rets.plot(subplots=True, grid=True, style='b', figsize=(8, 6))
# tag: es50_vs_rets
# title: Log returns of EURO STOXX 50 and VSTOXX


# In[79]:

xdat = rets['EUROSTOXX']
ydat = rets['VSTOXX']
model = pd.ols(y=ydat, x=xdat)
model


# In[80]:

model.beta


# In[81]:

plt.plot(xdat, ydat, 'r.')
ax = plt.axis()  # grab axis values
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, model.beta[1] + model.beta[0] * x, 'b', lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel('EURO STOXX 50 returns')
plt.ylabel('VSTOXX returns')
# tag: scatter_rets
# title: Scatter plot of log returns and regression line


# In[82]:

rets.corr()


# In[83]:

pd.rolling_corr(rets['EUROSTOXX'], rets['VSTOXX'],
                window=252).plot(grid=True, style='b')
# tag: roll_corr
# title: Rolling correlation between EURO STOXX 50 and VSTOXX


# ## High Frequency Data

# In[84]:

import numpy as np
import pandas as pd
import datetime as dt
from urllib import urlretrieve
get_ipython().magic('matplotlib inline')


# In[85]:

url1 = 'http://www.netfonds.no/quotes/posdump.php?'
url2 = 'date=%s%s%s&paper=NKE.N&csv_format=csv'
url = url1 + url2


# In[86]:

year = '2015'
month = '08'
days = ['03', '04', '05', '06', '07']
  # dates might need to be updated


# In[87]:

NKE = pd.DataFrame()
for day in days:
    NKE = NKE.append(pd.read_csv(url % (year, month, day),
                       index_col=0, header=0, parse_dates=True))
NKE.columns = ['bid', 'bdepth', 'bdeptht', 'offer', 'odepth', 'odeptht']
  # shorter colummn names


# In[88]:

NKE.info()


# In[89]:

NKE['bid'].plot(grid=True)
# tag: aapl
# title: Nike stock tick data for a week


# In[90]:

to_plot = NKE[['bid', 'bdeptht']][
    (NKE.index > dt.datetime(2015, 8, 4, 0, 0))
 &  (NKE.index < dt.datetime(2015, 8, 5, 2, 59))]
  # adjust dates to given data set
to_plot.plot(subplots=True, style='b', figsize=(8, 5), grid=True)
# tag: aapl_day
# title: Apple stock tick data and volume for a trading day


# In[91]:

NKE_resam = NKE.resample(rule='5min', how='mean')
np.round(NKE_resam.head(), 2)


# In[92]:

NKE_resam['bid'].fillna(method='ffill').plot(grid=True)
# tag: aapl_resam
# title: Resampled Apple stock tick data


# In[93]:

def reversal(x):
    return 2 * 95 - x


# In[94]:

NKE_resam['bid'].fillna(method='ffill').apply(reversal).plot(grid=True)
# tag: aapl_resam_apply
# title: Resampled Apple stock tick data with function applied to it


# In[95]:

get_ipython().system('rm ./data/*')
  # Windows: del /data/*


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
