
# coding: utf-8

# <img src="http://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0"><br>

# # Python for Finance

# **Analyze Big Financial Data**
# 
# O'Reilly (2014)
# 
# Yves Hilpisch

# <img style="border:1px solid grey;" src="http://hilpisch.com/python_for_finance.png" alt="Python for Finance" width="30%" align="left" border="0">

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

# ## Python

# In[1]:

import datetime as dt


# In[2]:

dt.datetime.now()


# In[3]:

to = dt.datetime.today()
to


# In[4]:

type(to)


# In[5]:

dt.datetime.today().weekday()
  # zero numbering 0 = Monday


# In[6]:

d = dt.datetime(2016, 10, 31, 10, 5, 30, 500000)
d


# In[7]:

print d


# In[8]:

str(d)


# In[9]:

d.year


# In[10]:

d.month


# In[11]:

d.day


# In[12]:

d.hour


# In[13]:

o = d.toordinal()
o


# In[14]:

dt.datetime.fromordinal(o)


# In[15]:

t = dt.datetime.time(d)
t


# In[16]:

type(t)


# In[17]:

dd = dt.datetime.date(d)
dd


# In[18]:

d.replace(second=0, microsecond=0)


# In[19]:

td = d - dt.datetime.now()
td


# In[20]:

type(td)


# In[21]:

td.days


# In[22]:

td.seconds


# In[23]:

td.microseconds


# In[24]:

td.total_seconds()


# In[25]:

d.isoformat()


# In[26]:

d.strftime("%A, %d. %B %Y %I:%M%p")


# In[27]:

dt.datetime.strptime('2017-03-31', '%Y-%m-%d')
  # year first and four digit year


# In[28]:

dt.datetime.strptime('30-4-16', '%d-%m-%y')
  # day first and two digit year


# In[29]:

ds = str(d)
ds


# In[30]:

dt.datetime.strptime(ds, '%Y-%m-%d %H:%M:%S.%f')


# In[31]:

dt.datetime.now()


# In[32]:

dt.datetime.utcnow()
  #  Universal Time, Coordinated


# In[33]:

dt.datetime.now() - dt.datetime.utcnow()
  # UTC + 2h = CET (summer)


# In[34]:

class UTC(dt.tzinfo):
    def utcoffset(self, d):
        return dt.timedelta(hours=0)
    def dst(self, d):
        return dt.timedelta(hours=0)
    def tzname(self, d):
        return "UTC"


# In[35]:

u = dt.datetime.utcnow()
u = u.replace(tzinfo=UTC())
  # attach time zone information
u


# In[36]:

class CET(dt.tzinfo):
    def utcoffset(self, d):
        return dt.timedelta(hours=2)
    def dst(self, d):
        return dt.timedelta(hours=1)
    def tzname(self, d):
        return "CET + 1"


# In[37]:

u.astimezone(CET())


# In[38]:

import pytz


# In[39]:

pytz.country_names['US']


# In[40]:

pytz.country_timezones['BE']


# In[41]:

pytz.common_timezones[-10:]


# In[42]:

u = dt.datetime.utcnow()
u = u.replace(tzinfo=pytz.utc)
u


# In[43]:

u.astimezone(pytz.timezone("CET"))


# In[44]:

u.astimezone(pytz.timezone("GMT"))


# In[45]:

u.astimezone(pytz.timezone("US/Central"))


# ## NumPy

# In[46]:

import numpy as np


# In[47]:

nd = np.datetime64('2015-10-31')
nd


# In[48]:

np.datetime_as_string(nd)


# In[49]:

np.datetime_data(nd)


# In[50]:

d


# In[51]:

nd = np.datetime64(d)
nd


# In[52]:

nd.astype(dt.datetime)


# In[53]:

nd = np.datetime64('2015-10', 'D')
nd


# In[54]:

np.datetime64('2015-10') == np.datetime64('2015-10-01')


# In[55]:

np.array(['2016-06-10', '2016-07-10', '2016-08-10'], dtype='datetime64')


# In[56]:

np.array(['2016-06-10T12:00:00', '2016-07-10T12:00:00',
          '2016-08-10T12:00:00'], dtype='datetime64[s]')


# In[57]:

np.arange('2016-01-01', '2016-01-04', dtype='datetime64')
  # daily frequency as default in this case


# In[58]:

np.arange('2016-01-01', '2016-10-01', dtype='datetime64[M]')
  # monthly frequency


# In[59]:

np.arange('2016-01-01', '2016-10-01', dtype='datetime64[W]')[:10]
  # weekly frequency


# In[60]:

dtl = np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
                dtype='datetime64[h]')
  # hourly frequency
dtl[:10]


# In[61]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[62]:

np.random.seed(3000)
rnd = np.random.standard_normal(len(dtl)).cumsum() ** 2


# In[63]:

fig = plt.figure()
plt.plot(dtl.astype(dt.datetime), rnd)
  # convert np.datetime to datetime.datetime
plt.grid(True)
fig.autofmt_xdate()
  # auto formatting of datetime xticks
# tag: datetime_plot
# title: Plot with datetime.datetime xticks auto-formatted


# In[64]:

np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
          dtype='datetime64[s]')[:10]
  # seconds as frequency


# In[65]:

np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
          dtype='datetime64[ms]')[:10]
  # milliseconds as frequency


# ## pandas

# In[66]:

import pandas as pd


# In[67]:

ts = pd.Timestamp('2016-06-30')
ts


# In[68]:

d = ts.to_datetime()
d


# In[69]:

pd.Timestamp(d)


# In[70]:

pd.Timestamp(nd)


# In[71]:

dti = pd.date_range('2016/01/01', freq='M', periods=12)
dti


# In[72]:

dti[6]


# In[73]:

pdi = dti.to_pydatetime()
pdi


# In[74]:

pd.DatetimeIndex(pdi)


# In[75]:

pd.DatetimeIndex(dtl.astype(pd.datetime))


# In[76]:

rnd = np.random.standard_normal(len(dti)).cumsum() ** 2


# In[77]:

df = pd.DataFrame(rnd, columns=['data'], index=dti)


# In[78]:

df.plot()
# tag: pandas_plot
# title: Pandas plot with Timestamp xticks auto-formatted


# In[79]:

pd.date_range('2016/01/01', freq='M', periods=12, tz=pytz.timezone('CET'))


# In[80]:

dti = pd.date_range('2016/01/01', freq='M', periods=12, tz='US/Eastern')
dti


# In[81]:

dti.tz_convert('GMT')


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
