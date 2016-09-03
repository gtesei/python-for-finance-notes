
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

# # Data Visualization

# ## Two-Dimensional Plotting

# In[1]:

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')
# import seaborn as sns; sns.set()
get_ipython().magic('matplotlib inline')


# ### One-Dimensional Data Set

# In[2]:

np.random.seed(1000)
y = np.random.standard_normal(20)


# In[3]:

x = range(len(y))
plt.plot(x, y)
# tag: matplotlib_0
# title: Plot given x- and y-values


# In[4]:

plt.plot(y)
# tag: matplotlib_1
# title: Plot given data as 1d-array


# In[5]:

plt.plot(y.cumsum())
# tag: matplotlib_2
# title: Plot given a 1d-array with method attached


# In[6]:

plt.plot(y.cumsum())
plt.grid(True)  # adds a grid
plt.axis('tight')  # adjusts the axis ranges
# tag: matplotlib_3_a
# title: Plot with grid and tight axes


# In[7]:

plt.plot(y.cumsum())
plt.grid(True)
plt.xlim(-1, 20)
plt.ylim(np.min(y.cumsum()) - 1,
         np.max(y.cumsum()) + 1)
# tag: matplotlib_3_b
# title: Plot with custom axes limits


# In[8]:

plt.figure(figsize=(7, 4))
  # the figsize parameter defines the
  # size of the figure in (width, height)
plt.plot(y.cumsum(), 'b', lw=1.5)
plt.plot(y.cumsum(), 'ro')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
# tag: matplotlib_4
# title: Plot with typical labels


# ### Two-Dimensional Data Set

# In[9]:

np.random.seed(2000)
y = np.random.standard_normal((20, 2)).cumsum(axis=0)


# In[10]:

plt.figure(figsize=(7, 4))
plt.plot(y, lw=1.5)
  # plots two lines
plt.plot(y, 'ro')
  # plots two dotted lines
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
# tag: matplotlib_5
# title: Plot with two data sets


# In[11]:

plt.figure(figsize=(7, 4))
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 1], lw=1.5, label='2nd')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
# tag: matplotlib_6
# title: Plot with labeled data sets


# In[12]:

y[:, 0] = y[:, 0] * 100
plt.figure(figsize=(7, 4))
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 1], lw=1.5, label='2nd')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')
# tag: matplotlib_7
# title: Plot with two differently scaled data sets


# In[13]:

fig, ax1 = plt.subplots()
plt.plot(y[:, 0], 'b', lw=1.5, label='1st')
plt.plot(y[:, 0], 'ro')
plt.grid(True)
plt.legend(loc=8)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value 1st')
plt.title('A Simple Plot')
ax2 = ax1.twinx()
plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
plt.plot(y[:, 1], 'ro')
plt.legend(loc=0)
plt.ylabel('value 2nd')
# tag: matplotlib_8
# title: Plot with two data sets and two y-axes


# In[14]:

plt.figure(figsize=(7, 5))
plt.subplot(211)
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 0], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.subplot(212)
plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
plt.plot(y[:, 1], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
# tag: matplotlib_9
# title: Plot with two sub-plots


# In[15]:

plt.figure(figsize=(9, 4))
plt.subplot(121)
plt.plot(y[:, 0], lw=1.5, label='1st')
plt.plot(y[:, 0], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('1st Data Set')
plt.subplot(122)
plt.bar(np.arange(len(y)), y[:, 1], width=0.5,
        color='g', label='2nd')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.title('2nd Data Set')
# tag: matplotlib_10
# title: Plot combining line/point sub-plot with bar sub-plot
# size: 80


# ### Other Plot Styles

# In[16]:

y = np.random.standard_normal((1000, 2))


# In[17]:

plt.figure(figsize=(7, 5))
plt.plot(y[:, 0], y[:, 1], 'ro')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
# tag: matplotlib_11_a
# title: Scatter plot via +plot+ function


# In[18]:

plt.figure(figsize=(7, 5))
plt.scatter(y[:, 0], y[:, 1], marker='o')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
# tag: matplotlib_11_b
# title: Scatter plot via +scatter+ function


# In[19]:

c = np.random.randint(0, 10, len(y))


# In[20]:

plt.figure(figsize=(7, 5))
plt.scatter(y[:, 0], y[:, 1], c=c, marker='o')
plt.colorbar()
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
# tag: matplotlib_11_c
# title: Scatter plot with third dimension


# In[21]:

plt.figure(figsize=(7, 4))
plt.hist(y, label=['1st', '2nd'], bins=25)
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Histogram')
# tag: matplotlib_12_a
# title: Histogram for two data sets


# In[22]:

plt.figure(figsize=(7, 4))
plt.hist(y, label=['1st', '2nd'], color=['b', 'g'],
            stacked=True, bins=20)
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Histogram')
# tag: matplotlib_12_b
# title: Stacked histogram for two data sets


# In[23]:

fig, ax = plt.subplots(figsize=(7, 4))
plt.boxplot(y)
plt.grid(True)
plt.setp(ax, xticklabels=['1st', '2nd'])
plt.xlabel('data set')
plt.ylabel('value')
plt.title('Boxplot')
# tag: matplotlib_13
# title: Boxplot for two data sets
# size: 70


# In[24]:

from matplotlib.patches import Polygon
def func(x):
    return 0.5 * np.exp(x) + 1

a, b = 0.5, 1.5  # integral limits
x = np.linspace(0, 2)
y = func(x)

fig, ax = plt.subplots(figsize=(7, 5))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)

# Illustrate the integral value, i.e. the area under the function
# between lower and upper limit
Ix = np.linspace(a, b)
Iy = func(Ix)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)

plt.text(0.5 * (a + b), 1, r"$\int_a^b f(x)\mathrm{d}x$",
         horizontalalignment='center', fontsize=20)

plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')

ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([func(a), func(b)])
ax.set_yticklabels(('$f(a)$', '$f(b)$'))
plt.grid(True)
# tag: matplotlib_math
# title: Exponential function, integral area and Latex labels
# size: 60


# ## Financial Plots

# In[25]:

import matplotlib.finance as mpf


# In[26]:

start = (2014, 5, 1)
end = (2014, 6, 30)

quotes = mpf.quotes_historical_yahoo('^GDAXI', start, end)


# In[27]:

quotes[:2]


# In[28]:

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(bottom=0.2)
mpf.candlestick(ax, quotes, width=0.6, colorup='b', colordown='r')
plt.grid(True)
ax.xaxis_date()
  # dates on the x axis
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=30)
# tag: matplotlib_14
# title: Candlestick chart for financial data
# size: 70


# In[29]:

fig, ax = plt.subplots(figsize=(8, 5))
mpf.plot_day_summary(ax, quotes, colorup='b', colordown='r')
plt.grid(True)
ax.xaxis_date()
plt.title('DAX Index')
plt.ylabel('index level')
plt.setp(plt.gca().get_xticklabels(), rotation=30)
# tag: matplotlib_15
# title: Daily summary chart for financial data
# size: 70


# In[30]:

quotes = np.array(mpf.quotes_historical_yahoo('YHOO', start, end))


# In[31]:

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 6))
mpf.candlestick(ax1, quotes, width=0.6, colorup='b', colordown='r')
ax1.set_title('Yahoo Inc.')
ax1.set_ylabel('index level')
ax1.grid(True)
ax1.xaxis_date()
plt.bar(quotes[:, 0] - 0.25, quotes[:, 5], width=0.5)
ax2.set_ylabel('volume')
ax2.grid(True)
ax2.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=30)
# tag: matplotlib_16
# title: Plot combining candlestick and volume bar chart
# size: 70


# ## 3d Plotting

# In[32]:

strike = np.linspace(50, 150, 24)
ttm = np.linspace(0.5, 2.5, 24)
strike, ttm = np.meshgrid(strike, ttm)


# In[33]:

strike[:2]


# In[34]:

iv = (strike - 100) ** 2 / (100 * strike) / ttm
  # generate fake implied volatilities


# In[35]:

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')

surf = ax.plot_surface(strike, ttm, iv, rstride=2, cstride=2,
                       cmap=plt.cm.coolwarm, linewidth=0.5,
                       antialiased=True)

ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')

fig.colorbar(surf, shrink=0.5, aspect=5)
# tag: matplotlib_17
# title: 3d surface plot for (fake) implied volatilities
# size: 70


# In[36]:

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 60)

ax.scatter(strike, ttm, iv, zdir='z', s=25,
           c='b', marker='^')

ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')

# tag: matplotlib_18
# title: 3d scatter plot for (fake) implied volatilities
# size: 70


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
