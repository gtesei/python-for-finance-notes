
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

# # Input-Output Operations

# ## Basic I/O with Python

# ### Writing Objects to Disk

# In[1]:

path = './data/'


# In[2]:

import numpy as np
from random import gauss


# In[3]:

a = [gauss(1.5, 2) for i in range(1000000)]
  # generation of normally distributed randoms


# In[4]:

import pickle


# In[5]:

pkl_file = open(path + 'data.pkl', 'w')
  # open file for writing
  # Note: existing file might be overwritten


# In[6]:

get_ipython().magic('time pickle.dump(a, pkl_file)')


# In[7]:

pkl_file


# In[8]:

pkl_file.close()


# In[9]:

ll $path*


# In[10]:

pkl_file = open(path + 'data.pkl', 'r')  # open file for reading


# In[11]:

get_ipython().magic('time b = pickle.load(pkl_file)')


# In[12]:

b[:5]


# In[13]:

a[:5]


# In[14]:

np.allclose(np.array(a), np.array(b))


# In[15]:

np.sum(np.array(a) - np.array(b))


# In[16]:

pkl_file = open(path + 'data.pkl', 'w')  # open file for writing


# In[17]:

get_ipython().magic('time pickle.dump(np.array(a), pkl_file)')


# In[18]:

get_ipython().magic('time pickle.dump(np.array(a) ** 2, pkl_file)')


# In[19]:

pkl_file.close()


# In[20]:

ll $path*


# In[21]:

pkl_file = open(path + 'data.pkl', 'r')  # open file for reading


# In[22]:

x = pickle.load(pkl_file)
x


# In[23]:

y = pickle.load(pkl_file)
y


# In[24]:

pkl_file.close()


# In[25]:

pkl_file = open(path + 'data.pkl', 'w')  # open file for writing
pickle.dump({'x' : x, 'y' : y}, pkl_file)
pkl_file.close()


# In[26]:

pkl_file = open(path + 'data.pkl', 'r')  # open file for writing
data = pickle.load(pkl_file)
pkl_file.close()
for key in data.keys():
    print key, data[key][:4]


# In[27]:

get_ipython().system('rm -f $path*')


# ### Reading and Writing Text Files

# In[28]:

rows = 5000
a = np.random.standard_normal((rows, 5))  # dummy data


# In[29]:

a.round(4)


# In[30]:

import pandas as pd
t = pd.date_range(start='2014/1/1', periods=rows, freq='H')
    # set of hourly datetime objects


# In[31]:

t


# In[32]:

csv_file = open(path + 'data.csv', 'w')  # open file for writing


# In[33]:

header = 'date,no1,no2,no3,no4,no5\n'
csv_file.write(header)


# In[34]:

for t_, (no1, no2, no3, no4, no5) in zip(t, a):
    s = '%s,%f,%f,%f,%f,%f\n' % (t_, no1, no2, no3, no4, no5)
    csv_file.write(s)
csv_file.close()


# In[35]:

ll $path*


# In[36]:

csv_file = open(path + 'data.csv', 'r')  # open file for reading


# In[37]:

for i in range(5):
    print csv_file.readline(),


# In[38]:

csv_file = open(path + 'data.csv', 'r')
content = csv_file.readlines()
for line in content[:5]:
    print line,


# In[39]:

csv_file.close()
get_ipython().system('rm -f $path*')


# ### SQL Databases

# In[40]:

import sqlite3 as sq3


# In[41]:

query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)'


# In[42]:

con = sq3.connect(path + 'numbs.db')


# In[43]:

con.execute(query)


# In[44]:

con.commit()


# In[45]:

import datetime as dt


# In[46]:

con.execute('INSERT INTO numbs VALUES(?, ?, ?)',
            (dt.datetime.now(), 0.12, 7.3))


# In[47]:

data = np.random.standard_normal((10000, 2)).round(5)


# In[48]:

for row in data:
    con.execute('INSERT INTO numbs VALUES(?, ?, ?)',
                (dt.datetime.now(), row[0], row[1]))
con.commit()


# In[49]:

con.execute('SELECT * FROM numbs').fetchmany(10)


# In[50]:

pointer = con.execute('SELECT * FROM numbs')


# In[51]:

for i in range(3):
    print pointer.fetchone()


# In[52]:

con.close()
get_ipython().system('rm -f $path*')


# ### Writing and Reading Numpy Arrays

# In[53]:

import numpy as np


# In[54]:

dtimes = np.arange('2015-01-01 10:00:00', '2021-12-31 22:00:00',
                  dtype='datetime64[m]')  # minute intervals
len(dtimes)


# In[55]:

dty = np.dtype([('Date', 'datetime64[m]'), ('No1', 'f'), ('No2', 'f')])
data = np.zeros(len(dtimes), dtype=dty)


# In[56]:

data['Date'] = dtimes


# In[57]:

a = np.random.standard_normal((len(dtimes), 2)).round(5)
data['No1'] = a[:, 0]
data['No2'] = a[:, 1]


# In[58]:

get_ipython().magic("time np.save(path + 'array', data)  # suffix .npy is added")


# In[59]:

ll $path*


# In[60]:

get_ipython().magic("time np.load(path + 'array.npy')")


# In[61]:

data = np.random.standard_normal((10000, 6000))


# In[62]:

get_ipython().magic("time np.save(path + 'array', data)")


# In[63]:

ll $path*


# In[64]:

get_ipython().magic("time np.load(path + 'array.npy')")


# In[65]:

data = 0.0
get_ipython().system('rm -f $path*')


# ## I/O with pandas

# In[66]:

import numpy as np
import pandas as pd
data = np.random.standard_normal((1000000, 5)).round(5)
        # sample data set


# In[67]:

filename = path + 'numbs'


# ### SQL Database

# In[68]:

import sqlite3 as sq3


# In[69]:

query = 'CREATE TABLE numbers (No1 real, No2 real,        No3 real, No4 real, No5 real)'


# In[70]:

con = sq3.Connection(filename + '.db')


# In[71]:

con.execute(query)


# In[72]:

get_ipython().run_cell_magic('time', '', "con.executemany('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', data)\ncon.commit()")


# In[73]:

ll $path*


# In[74]:

get_ipython().run_cell_magic('time', '', "temp = con.execute('SELECT * FROM numbers').fetchall()\nprint temp[:2]\ntemp = 0.0")


# In[75]:

get_ipython().run_cell_magic('time', '', "query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0'\nres = np.array(con.execute(query).fetchall()).round(3)")


# In[76]:

res = res[::100]  # every 100th result
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(res[:, 0], res[:, 1], 'ro')
plt.grid(True); plt.xlim(-0.5, 4.5); plt.ylim(-4.5, 0.5)
# tag: scatter_query
# title: Plot of the query result
# size: 60


# ### From SQL to pandas

# In[77]:

import pandas.io.sql as pds


# In[78]:

get_ipython().magic("time data = pds.read_sql('SELECT * FROM numbers', con)")


# In[79]:

data.head()


# In[80]:

get_ipython().magic("time data[(data['No1'] > 0) & (data['No2'] < 0)].head()")


# In[81]:

get_ipython().run_cell_magic('time', '', "res = data[['No1', 'No2']][((data['No1'] > 0.5) | (data['No1'] < -0.5))\n                     & ((data['No2'] < -1) | (data['No2'] > 1))]")


# In[82]:

plt.plot(res.No1, res.No2, 'ro')
plt.grid(True); plt.axis('tight')
# tag: data_scatter_1
# title: Scatter plot of complex query results
# size: 55


# In[83]:

h5s = pd.HDFStore(filename + '.h5s', 'w')


# In[84]:

get_ipython().magic("time h5s['data'] = data")


# In[85]:

h5s


# In[86]:

h5s.close()


# In[87]:

get_ipython().run_cell_magic('time', '', "h5s = pd.HDFStore(filename + '.h5s', 'r')\ntemp = h5s['data']\nh5s.close()")


# In[88]:

np.allclose(np.array(temp), np.array(data))


# In[89]:

temp = 0.0


# In[90]:

ll $path*


# ### Data as CSV File

# In[91]:

get_ipython().magic("time data.to_csv(filename + '.csv')")


# In[92]:

get_ipython().run_cell_magic('time', '', "pd.read_csv(filename + '.csv')[['No1', 'No2',\n                                'No3', 'No4']].hist(bins=20)\n# tag: data_hist_3\n# title: Histogram of 4 data sets\n# size: 60")


# ### Data as Excel File

# In[93]:

get_ipython().magic("time data[:100000].to_excel(filename + '.xlsx')")


# In[94]:

get_ipython().magic("time pd.read_excel(filename + '.xlsx', 'Sheet1').cumsum().plot()")
# tag: data_paths
# title: Paths of random data from Excel file
# size: 60


# In[95]:

ll $path*


# In[96]:

rm -f $path*


# ## Fast I/O with PyTables

# In[97]:

import numpy as np
import tables as tb
import datetime as dt
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ### Working with Tables

# In[98]:

filename = path + 'tab.h5'
h5 = tb.open_file(filename, 'w') 


# In[99]:

rows = 2000000


# In[100]:

row_des = {
    'Date': tb.StringCol(26, pos=1),
    'No1': tb.IntCol(pos=2),
    'No2': tb.IntCol(pos=3),
    'No3': tb.Float64Col(pos=4),
    'No4': tb.Float64Col(pos=5)
    }


# In[101]:

filters = tb.Filters(complevel=0)  # no compression
tab = h5.create_table('/', 'ints_floats', row_des,
                      title='Integers and Floats',
                      expectedrows=rows, filters=filters)


# In[102]:

tab


# In[103]:

pointer = tab.row


# In[104]:

ran_int = np.random.randint(0, 10000, size=(rows, 2))
ran_flo = np.random.standard_normal((rows, 2)).round(5)


# In[105]:

get_ipython().run_cell_magic('time', '', "for i in range(rows):\n    pointer['Date'] = dt.datetime.now()\n    pointer['No1'] = ran_int[i, 0]\n    pointer['No2'] = ran_int[i, 1] \n    pointer['No3'] = ran_flo[i, 0]\n    pointer['No4'] = ran_flo[i, 1] \n    pointer.append()\n      # this appends the data and\n      # moves the pointer one row forward\ntab.flush()")


# In[106]:

tab


# In[107]:

ll $path*


# In[108]:

dty = np.dtype([('Date', 'S26'), ('No1', '<i4'), ('No2', '<i4'),
                                 ('No3', '<f8'), ('No4', '<f8')])
sarray = np.zeros(len(ran_int), dtype=dty)


# In[109]:

sarray


# In[110]:

get_ipython().run_cell_magic('time', '', "sarray['Date'] = dt.datetime.now()\nsarray['No1'] = ran_int[:, 0]\nsarray['No2'] = ran_int[:, 1]\nsarray['No3'] = ran_flo[:, 0]\nsarray['No4'] = ran_flo[:, 1]")


# In[111]:

get_ipython().run_cell_magic('time', '', "h5.create_table('/', 'ints_floats_from_array', sarray,\n                      title='Integers and Floats',\n                      expectedrows=rows, filters=filters)")


# In[112]:

h5


# In[113]:

h5.remove_node('/', 'ints_floats_from_array')


# In[114]:

tab[:3]


# In[115]:

tab[:4]['No4']


# In[116]:

get_ipython().magic("time np.sum(tab[:]['No3'])")


# In[117]:

get_ipython().magic("time np.sum(np.sqrt(tab[:]['No1']))")


# In[118]:

get_ipython().run_cell_magic('time', '', "plt.hist(tab[:]['No3'], bins=30)\nplt.grid(True)\nprint len(tab[:]['No3'])\n# tag: data_hist\n# title: Histogram of data\n# size: 60")


# In[119]:

get_ipython().run_cell_magic('time', '', "res = np.array([(row['No3'], row['No4']) for row in\n        tab.where('((No3 < -0.5) | (No3 > 0.5)) \\\n                 & ((No4 < -1) | (No4 > 1))')])[::100]")


# In[120]:

plt.plot(res.T[0], res.T[1], 'ro')
plt.grid(True)
# tag: scatter_data
# title: Scatter plot of query result
# size: 70


# In[121]:

get_ipython().run_cell_magic('time', '', 'values = tab.cols.No3[:]\nprint "Max %18.3f" % values.max()\nprint "Ave %18.3f" % values.mean()\nprint "Min %18.3f" % values.min()\nprint "Std %18.3f" % values.std()')


# In[122]:

get_ipython().run_cell_magic('time', '', "results = [(row['No1'], row['No2']) for row in\n           tab.where('((No1 > 9800) | (No1 < 200)) \\\n                    & ((No2 > 4500) & (No2 < 5500))')]\nfor res in results[:4]:\n    print res")


# In[123]:

get_ipython().run_cell_magic('time', '', "results = [(row['No1'], row['No2']) for row in\n           tab.where('(No1 == 1234) & (No2 > 9776)')]\nfor res in results:\n    print res")


# ### Working with Compressed Tables

# In[124]:

filename = path + 'tab.h5c'
h5c = tb.open_file(filename, 'w') 


# In[125]:

filters = tb.Filters(complevel=4, complib='blosc')


# In[126]:

tabc = h5c.create_table('/', 'ints_floats', sarray,
                        title='Integers and Floats',
                      expectedrows=rows, filters=filters)


# In[127]:

get_ipython().run_cell_magic('time', '', "res = np.array([(row['No3'], row['No4']) for row in\n             tabc.where('((No3 < -0.5) | (No3 > 0.5)) \\\n                       & ((No4 < -1) | (No4 > 1))')])[::100]")


# In[128]:

get_ipython().magic('time arr_non = tab.read()')


# In[129]:

get_ipython().magic('time arr_com = tabc.read()')


# In[130]:

ll $path*


# In[131]:

h5c.close()


# ### Working with Arrays

# In[132]:

get_ipython().run_cell_magic('time', '', "arr_int = h5.create_array('/', 'integers', ran_int)\narr_flo = h5.create_array('/', 'floats', ran_flo)")


# In[133]:

h5


# In[134]:

ll $path*


# In[135]:

h5.close()


# In[136]:

get_ipython().system('rm -f $path*')


# ### Out-of-Memory Computations

# In[137]:

filename = path + 'array.h5'
h5 = tb.open_file(filename, 'w') 


# In[138]:

n = 100
ear = h5.createEArray(h5.root, 'ear',
                      atom=tb.Float64Atom(),
                      shape=(0, n))


# In[139]:

get_ipython().run_cell_magic('time', '', 'rand = np.random.standard_normal((n, n))\nfor i in range(750):\n    ear.append(rand)\near.flush()')


# In[140]:

ear


# In[141]:

ear.size_on_disk


# In[142]:

out = h5.createEArray(h5.root, 'out',
                      atom=tb.Float64Atom(),
                      shape=(0, n))


# In[143]:

expr = tb.Expr('3 * sin(ear) + sqrt(abs(ear))')
  # the numerical expression as a string object
expr.setOutput(out, append_mode=True)
  # target to store results is disk-based array


# In[144]:

get_ipython().magic('time expr.eval()')
  # evaluation of the numerical expression
  # and storage of results in disk-based array


# In[145]:

out[0, :10]


# In[146]:

get_ipython().magic('time imarray = ear.read()')
  # read whole array into memory


# In[147]:

import numexpr as ne
expr = '3 * sin(imarray) + sqrt(abs(imarray))'


# In[148]:

ne.set_num_threads(16)
get_ipython().magic('time ne.evaluate(expr)[0, :10]')


# In[149]:

h5.close()


# In[150]:

get_ipython().system('rm -f $path*')


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
