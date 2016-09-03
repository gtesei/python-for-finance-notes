
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

# # Web Integration

# ## Web Basics

# ### ftplib

# In[1]:

import ftplib
import numpy as np


# In[2]:

ftp = ftplib.FTP('quant-platform.com')


# In[3]:

ftp.login(user='python', passwd='python')


# In[4]:

np.save('./data/array', np.random.standard_normal((100, 100)))


# In[5]:

f = open('./data/array.npy', 'r')


# In[6]:

ftp.storbinary('STOR array.npy', f)


# In[7]:

ftp.retrlines('LIST')


# In[8]:

f = open('./data/array_ftp.npy', 'wb').write


# In[9]:

ftp.retrbinary('RETR array.npy', f)


# In[10]:

ftp.delete('array.npy')


# In[11]:

ftp.retrlines('LIST')


# In[12]:

ftp.close()


# In[13]:

get_ipython().system('ls -n ./data')


# In[14]:

get_ipython().system('rm -f ./data/arr*')
  # clean-up directory


# In[15]:

ftps = ftplib.FTP_TLS('quant-platform.com')


# In[16]:

ftps.login(user='python', passwd='python')


# In[17]:

ftps.prot_p()


# In[18]:

ftps.retrlines('LIST')


# In[19]:

ftps.close()


# ### httplib

# In[20]:

import httplib


# In[21]:

http = httplib.HTTPConnection('hilpisch.com')


# In[22]:

http.request('GET', '/index.htm')


# In[23]:

resp = http.getresponse()


# In[24]:

resp.status, resp.reason


# In[25]:

content = resp.read()
content[:100]
  # first 100 characters of the file


# In[26]:

index = content.find(' E ')
index


# In[27]:

content[index:index + 29]


# In[28]:

http.close()


# ### urllib

# In[29]:

import urllib


# In[30]:

url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
url += '&s=YHOO&a=01&b=1&c=2014&d=02&e=6&f=2014'


# In[31]:

connect = urllib.urlopen(url)


# In[32]:

data = connect.read()


# In[33]:

print data


# In[34]:

url = 'http://ichart.finance.yahoo.com/table.csv?g=d&ignore=.csv'
url += '&%s'  # for replacement with parameters
url += '&d=06&e=30&f=2014'


# In[35]:

params = urllib.urlencode({'s': 'MSFT', 'a': '05', 'b': 1, 'c': 2014})


# In[36]:

params


# In[37]:

url % params


# In[38]:

connect = urllib.urlopen(url % params)


# In[39]:

data = connect.read()


# In[40]:

print data


# In[41]:

urllib.urlretrieve(url % params, './data/msft.csv')


# In[42]:

csv = open('./data/msft.csv', 'r')
csv.readlines()[:5]


# In[43]:

get_ipython().system('rm -f ./data/*')


# ## Web Plotting

# ### Static Plots

# In[44]:

import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[45]:

url = 'http://ichart.yahoo.com/table.csv?s=MSFT&a=0&b=1&c=2009'
data = pd.read_csv(url, parse_dates=['Date'])


# In[46]:

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
data.plot(x='Date', y='Close', grid=True, ax=ax)
# tag: microsoft
# title: Historical stock prices for Microsoft since January 2009 (+matplotlib+)


# ### Interactive Plots

# **REMARK**: The original version used Bokeh for Web plotting. Plotly seems to be the more easy and intuitive way for generating interactive D3.js Web plots.

# In[47]:

import plotly.plotly as py
import cufflinks as cf
py.sign_in('Python-Demo-Account', 'gwt101uhh0')


# In[48]:

# transforming the above mpl figure
# to interactive D3.js plot
py.iplot_mpl(fig)


# In[49]:

# direct approach with Cufflinks
data.set_index('Date')['Close'].iplot(world_readable=True)


# ### Real-Time Plots

# #### Real-Time FX Data

# In[50]:

import time
import pandas as pd
import datetime as dt
import requests


# In[51]:

url = 'http://api-sandbox.oanda.com/v1/prices?instruments=%s'
  # real-time FX (dummy!) data from JSON API


# In[52]:

instrument = 'EUR_USD'
api = requests.get(url % instrument)


# In[53]:

data = api.json()
data


# In[54]:

data = data['prices'][0]
data


# In[55]:

ticks = pd.DataFrame({'bid': data['bid'],
                      'ask': data['ask'],
                      'instrument': data['instrument'],
                      'time': pd.Timestamp(data['time'])},
                      index=[pd.Timestamp(data['time']),])
  # initialization of ticks DataFrame


# In[56]:

ticks[['ask', 'bid', 'instrument']]


# #### Real-Time Stock Price Quotes

# In[57]:

url1 = 'http://www.netfonds.no/quotes/posdump.php?'
url2 = 'date=%s%s%s&paper=%s.N&csv_format=csv'
url = url1 + url2


# In[58]:

# must be a business day
today = dt.datetime.now()
y = '%d' % today.year
  # current year
m = '%02d' % today.month
  # current month, add leading zero if needed
d = '%02d' % today.day
  # current day, add leading zero if needed
sym = 'NKE'
  # Nike Inc. stocks


# In[59]:

y, m, d, sym


# In[60]:

urlreq = url % (y, m, d, sym)
urlreq


# In[61]:

data = pd.read_csv(urlreq, parse_dates=['time'])
  # initialize DataFrame object


# In[62]:

data.info()


# ## Rapid Web Applications

# ### Traders' Chat Room

# ### Data Modeling

# ### The Python Code

# #### Imports and Database Preliminaries

# #### Core Functionality

# ### Templating

# In[63]:

'%d, %d, %d' % (1, 2, 3)


# In[64]:

'{}, {}, {}'.format(1, 2, 3)


# In[65]:

'{}, {}, {}'.format(*'123')


# In[66]:

templ = '''<!doctype html>
  Just print out <b>numbers</b> provided to the template.
  <br><br>
  {% for number in numbers %}
    {{ number }}
  {% endfor %}
'''


# In[67]:

from jinja2 import Template


# In[68]:

t = Template(templ)


# In[69]:

html = t.render(numbers=range(5))


# In[70]:

html


# In[71]:

from IPython.display import HTML
HTML(html)


# ### Styling

# In[72]:

import os
for path, dirs, files in os.walk('../python/tradechat'):
  print path
  for f in files:
    print f


# ## Web Services

# ### The Financial Model

# ### The Implementation

# In[73]:

import sys
sys.path.append("../python/volservice")
  # adjust if necessary to your path


# In[74]:

from werkzeug.wrappers import Request, Response 


# In[75]:

from vol_pricing_service import get_option_value


# In[76]:

def application(environ, start_response):
    request = Request(environ)
      # wrap environ in new object
    text = get_option_value(request.args)
      # provide all paramters of the call to function
      # get back either error message or option value
    response = Response(text, mimetype='text/html')
      # generate response object based on the returned text
    return response(environ, start_response)


# In[77]:

import numpy as np
import urllib
url = 'http://localhost:4000/'


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
