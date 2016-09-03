
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

# # Data Types and Structures

# ## Basic Data Types

# ### Integers

# In[1]:

a = 10
type(a)


# In[2]:

a.bit_length()


# In[3]:

a = 100000
a.bit_length()


# In[4]:

googol = 10 ** 100
googol


# In[5]:

googol.bit_length()


# In[6]:

1 + 4


# In[7]:

1 / 4


# In[8]:

type(1 / 4)


# ### Floats

# In[9]:

1. / 4


# In[10]:

type (1. / 4)


# In[11]:

b = 0.35
type(b)


# In[12]:

b + 0.1


# In[13]:

c = 0.5
c.as_integer_ratio()


# In[14]:

b.as_integer_ratio()


# In[15]:

import decimal
from decimal import Decimal


# In[16]:

decimal.getcontext()


# In[17]:

d = Decimal(1) / Decimal (11)
d


# In[18]:

decimal.getcontext().prec = 4  # lower precision than default


# In[19]:

e = Decimal(1) / Decimal (11)
e


# In[20]:

decimal.getcontext().prec = 50  # higher precision than default


# In[21]:

f = Decimal(1) / Decimal (11)
f


# In[22]:

g = d + e + f
g


# ### Strings

# In[23]:

t = 'this is a string object'


# In[24]:

t.capitalize()


# In[25]:

t.split()


# In[26]:

t.find('string')


# In[27]:

t.find('Python')


# In[28]:

t.replace(' ', '|')


# In[29]:

'http://www.python.org'.strip('htp:/')


# In[30]:

import re


# In[31]:

series = """
'01/18/2014 13:00:00', 100, '1st';
'01/18/2014 13:30:00', 110, '2nd';
'01/18/2014 14:00:00', 120, '3rd'
"""


# In[32]:

dt = re.compile("'[0-9/:\s]+'")  # datetime


# In[33]:

result = dt.findall(series)
result


# In[34]:

from datetime import datetime
pydt = datetime.strptime(result[0].replace("'", ""),
                         '%m/%d/%Y %H:%M:%S')
pydt


# In[35]:

print pydt


# In[36]:

print type(pydt)


# ## Basic Data Structures

# ### Tuples

# In[37]:

t = (1, 2.5, 'data')
type(t)


# In[38]:

t = 1, 2.5, 'data'
type(t)


# In[39]:

t[2]


# In[40]:

type(t[2])


# In[41]:

t.count('data')


# In[42]:

t.index(1)


# ### Lists

# In[43]:

l = [1, 2.5, 'data']
l[2]


# In[44]:

l = list(t)
l


# In[45]:

type(l)


# In[46]:

l.append([4, 3])  # append list at the end
l


# In[47]:

l.extend([1.0, 1.5, 2.0])  # append elements of list
l


# In[48]:

l.insert(1, 'insert')  # insert object before index position
l


# In[49]:

l.remove('data')  # remove first occurence of object
l


# In[50]:

p = l.pop(3)  # removes and returns object at index
print l, p


# In[51]:

l[2:5]  # 3rd to 5th element


# ### Excursion: Control Structures

# In[52]:

for element in l[2:5]:
    print element ** 2


# In[53]:

r = range(0, 8, 1)  # start, end, step width
r


# In[54]:

type(r)


# In[55]:

for i in range(2, 5):
    print l[i] ** 2


# In[56]:

for i in range(1, 10):
    if i % 2 == 0:  # % is for modulo
        print "%d is even" % i
    elif i % 3 == 0:
        print "%d is multiple of 3" % i
    else:
        print "%d is odd" % i


# In[57]:

total = 0
while total < 100:
    total += 1
print total


# In[58]:

m = [i ** 2 for i in range(5)]
m


# ### Excursion: Functional Programming

# In[59]:

def f(x):
    return x ** 2
f(2)


# In[60]:

def even(x):
    return x % 2 == 0
even(3)


# In[61]:

map(even, range(10))


# In[62]:

map(lambda x: x ** 2, range(10))


# In[63]:

filter(even, range(15)) 


# In[64]:

reduce(lambda x, y: x + y, range(10))


# In[65]:

def cumsum(l):
    total = 0
    for elem in l:
        total += elem
    return total
cumsum(range(10))


# ### Dicts

# In[66]:

d = {
     'Name' : 'Angela Merkel',
     'Country' : 'Germany',
     'Profession' : 'Chancelor',
     'Age' : 60
     }
type(d)


# In[67]:

print d['Name'], d['Age']


# In[68]:

d.keys()


# In[69]:

d.values()


# In[70]:

d.items()


# In[71]:

birthday = True
if birthday is True:
    d['Age'] += 1
print d['Age']


# In[72]:

for item in d.iteritems():
    print item


# In[73]:

for value in d.itervalues():
    print type(value)


# ### Sets

# In[74]:

s = set(['u', 'd', 'ud', 'du', 'd', 'du'])
s


# In[75]:

t = set(['d', 'dd', 'uu', 'u'])


# In[76]:

s.union(t)  # all of s and t


# In[77]:

s.intersection(t)  # both in s and t


# In[78]:

s.difference(t)  # in s but not t


# In[79]:

t.difference(s)  # in t but not s


# In[80]:

s.symmetric_difference(t)  # in either one but not both


# In[81]:

from random import randint
l = [randint(0, 10) for i in range(1000)]
    # 1,000 random integers between 0 and 10
len(l)  # number of elements in l


# In[82]:

l[:20]


# In[83]:

s = set(l)
s


# ## NumPy Data Structures

# ### Arrays with Python Lists

# In[84]:

v = [0.5, 0.75, 1.0, 1.5, 2.0]  # vector of numbers


# In[85]:

m = [v, v, v]  # matrix of numbers
m


# In[86]:

m[1]


# In[87]:

m[1][0]


# In[88]:

v1 = [0.5, 1.5]
v2 = [1, 2]
m = [v1, v2]
c = [m, m]  # cube of numbers
c


# In[89]:

c[1][1][0]


# In[90]:

v = [0.5, 0.75, 1.0, 1.5, 2.0]
m = [v, v, v]
m


# In[91]:

v[0] = 'Python'
m


# In[92]:

from copy import deepcopy
v = [0.5, 0.75, 1.0, 1.5, 2.0]
m = 3 * [deepcopy(v), ]
m


# In[93]:

v[0] = 'Python'
m


# ### Regular NumPy Arrays

# In[94]:

import numpy as np


# In[95]:

a = np.array([0, 0.5, 1.0, 1.5, 2.0])
type(a)


# In[96]:

a[:2]  # indexing as with list objects in 1 dimension


# In[97]:

a.sum()  # sum of all elements


# In[98]:

a.std()  # standard deviation


# In[99]:

a.cumsum()  # running cumulative sum


# In[100]:

a * 2


# In[101]:

a ** 2


# In[102]:

np.sqrt(a)


# In[103]:

b = np.array([a, a * 2])
b


# In[104]:

b[0]  # first row


# In[105]:

b[0, 2]  # third element of first row


# In[106]:

b.sum()


# In[107]:

b.sum(axis=0)
  # sum along axis 0, i.e. column-wise sum


# In[108]:

b.sum(axis=1)
  # sum along axis 1, i.e. row-wise sum


# In[109]:

c = np.zeros((2, 3, 4), dtype='i', order='C')  # also: np.ones()
c


# In[110]:

d = np.ones_like(c, dtype='f16', order='C')  # also: np.zeros_like()
d


# In[111]:

import random
I = 5000 


# In[112]:

get_ipython().magic('time mat = [[random.gauss(0, 1) for j in range(I)] for i in range(I)]')
  # a nested list comprehension


# In[113]:

get_ipython().magic('time reduce(lambda x, y: x + y,           [reduce(lambda x, y: x + y, row)              for row in mat])')


# In[114]:

get_ipython().magic('time mat = np.random.standard_normal((I, I))')


# In[115]:

get_ipython().magic('time mat.sum()')


# ### Structured Arrays

# In[116]:

dt = np.dtype([('Name', 'S10'), ('Age', 'i4'),
               ('Height', 'f'), ('Children/Pets', 'i4', 2)])
s = np.array([('Smith', 45, 1.83, (0, 1)),
              ('Jones', 53, 1.72, (2, 2))], dtype=dt)
s


# In[117]:

s['Name']


# In[118]:

s['Height'].mean()


# In[119]:

s[1]['Age']


# ## Vectorization of Code

# ### Basic Vectorization

# In[120]:

r = np.random.standard_normal((4, 3))
s = np.random.standard_normal((4, 3))


# In[121]:

r + s


# In[122]:

2 * r + 3


# In[123]:

s = np.random.standard_normal(3)
r + s


# In[124]:

# causes intentional error
# s = np.random.standard_normal(4)
# r + s


# In[125]:

# r.transpose() + s


# In[126]:

np.shape(r.T)


# In[127]:

def f(x):
    return 3 * x + 5


# In[128]:

f(0.5)  # float object


# In[129]:

f(r)  # NumPy array


# In[130]:

# causes intentional error
# import math
# math.sin(r)


# In[131]:

np.sin(r)  # array as input


# In[132]:

np.sin(np.pi)  # float as input


# ### Memory Layout

# In[133]:

x = np.random.standard_normal((5, 10000000))
y = 2 * x + 3  # linear equation y = a * x + b
C = np.array((x, y), order='C')
F = np.array((x, y), order='F')
x = 0.0; y = 0.0  # memory clean-up


# In[134]:

C[:2].round(2)


# In[135]:

get_ipython().magic('timeit C.sum()')


# In[136]:

get_ipython().magic('timeit F.sum()')


# In[137]:

get_ipython().magic('timeit C[0].sum(axis=0)')


# In[138]:

get_ipython().magic('timeit C[0].sum(axis=1)')


# In[139]:

get_ipython().magic('timeit F.sum(axis=0)')


# In[140]:

get_ipython().magic('timeit F.sum(axis=1)')


# In[141]:

F = 0.0; C = 0.0  # memory clean-up


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
