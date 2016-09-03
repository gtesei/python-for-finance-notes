
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

# # Mathematical Tools

# ## Approximation

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

def f(x):
    return np.sin(x) + 0.5 * x


# In[3]:

x = np.linspace(-2 * np.pi, 2 * np.pi, 50)


# In[4]:

plt.plot(x, f(x), 'b')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot
# title: Example function plot
# size: 60


# ### Regression

# #### Monomials as Basis Functions

# In[5]:

reg = np.polyfit(x, f(x), deg=1)
ry = np.polyval(reg, x)


# In[6]:

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_reg_1
# title: Example function and linear regression
# size: 60


# In[7]:

reg = np.polyfit(x, f(x), deg=5)
ry = np.polyval(reg, x)


# In[8]:

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_reg_2
# title: Regression with monomials up to order 5
# size: 60


# In[9]:

reg = np.polyfit(x, f(x), 7)
ry = np.polyval(reg, x)


# In[10]:

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_reg_3
# title: Regression with monomials up to order 7
# size: 60


# In[11]:

np.allclose(f(x), ry)


# In[12]:

np.sum((f(x) - ry) ** 2) / len(x)


# #### Individual Basis Functions

# In[13]:

matrix = np.zeros((3 + 1, len(x)))
matrix[3, :] = x ** 3
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1


# In[14]:

reg = np.linalg.lstsq(matrix.T, f(x))[0]


# In[15]:

reg


# In[16]:

ry = np.dot(reg, matrix)


# In[17]:

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_reg_4
# title: Regression via least-squares function
# size: 60


# In[18]:

matrix[3, :] = np.sin(x)
reg = np.linalg.lstsq(matrix.T, f(x))[0]
ry = np.dot(reg, matrix)


# In[19]:

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_reg_5
# title: Regression using individual functions
# size: 60


# In[20]:

np.allclose(f(x), ry)


# In[21]:

np.sum((f(x) - ry) ** 2) / len(x)


# In[22]:

reg


# #### Noisy Data

# In[23]:

xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
xn = xn + 0.15 * np.random.standard_normal(len(xn))
yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))


# In[24]:

reg = np.polyfit(xn, yn, 7)
ry = np.polyval(reg, xn)


# In[25]:

plt.plot(xn, yn, 'b^', label='f(x)')
plt.plot(xn, ry, 'ro', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_reg_6
# title: Regression with noisy data
# size: 60


# #### Unsorted Data

# In[26]:

xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
yu = f(xu)


# In[27]:

print xu[:10].round(2)
print yu[:10].round(2)


# In[28]:

reg = np.polyfit(xu, yu, 5)
ry = np.polyval(reg, xu)


# In[29]:

plt.plot(xu, yu, 'b^', label='f(x)')
plt.plot(xu, ry, 'ro', label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_reg_7
# title: Regression with unsorted data
# size: 60


# #### Multiple Dimensions

# In[30]:

def fm((x, y)):
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2


# In[31]:

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)
  # generates 2-d grids out of the 1-d arrays
Z = fm((X, Y))
x = X.flatten()
y = Y.flatten()
  # yields 1-d arrays from the 2-d grids


# In[32]:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm,
        linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
# tag: sin_plot_3d_1
# title: Function with two parameters
# size: 60


# In[33]:

matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1


# In[34]:

import statsmodels.api as sm


# In[35]:

model = sm.OLS(fm((x, y)), matrix).fit()


# In[36]:

model.rsquared


# In[37]:

a = model.params
a


# In[38]:

def reg_func(a, (x, y)):
    f6 = a[6] * np.sqrt(y)
    f5 = a[5] * np.sin(x)
    f4 = a[4] * y ** 2
    f3 = a[3] * x ** 2
    f2 = a[2] * y
    f1 = a[1] * x
    f0 = a[0] * 1
    return (f6 + f5 + f4 + f3 +
            f2 + f1 + f0)


# In[39]:

RZ = reg_func(a, (X, Y))


# In[40]:

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
            cmap=mpl.cm.coolwarm, linewidth=0.5,
            antialiased=True)
surf2 = ax.plot_wireframe(X, Y, RZ, rstride=2, cstride=2,
                          label='regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=5)
# tag: sin_plot_3d_2
# title: Higher dimension regression
# size: 60


# ### Interpolation

# In[41]:

import scipy.interpolate as spi


# In[42]:

x = np.linspace(-2 * np.pi, 2 * np.pi, 25)


# In[43]:

def f(x):
    return np.sin(x) + 0.5 * x


# In[44]:

ipo = spi.splrep(x, f(x), k=1)


# In[45]:

iy = spi.splev(x, ipo)


# In[46]:

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, iy, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_ipo_1
# title: Example plot with linear interpolation
# size: 60


# In[47]:

np.allclose(f(x), iy)


# In[48]:

xd = np.linspace(1.0, 3.0, 50)
iyd = spi.splev(xd, ipo)


# In[49]:

plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_ipo_2
# title: Example plot (detail) with linear interpolation
# size: 60


# In[50]:

ipo = spi.splrep(x, f(x), k=3)
iyd = spi.splev(xd, ipo)


# In[51]:

plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
# tag: sin_plot_ipo_3
# title: Example plot (detail) with cubic splines interpolation
# size: 60


# In[52]:

np.allclose(f(xd), iyd)


# In[53]:

np.sum((f(xd) - iyd) ** 2) / len(xd)


# ## Convex Optimization

# In[54]:

def fm((x, y)):
    return (np.sin(x) + 0.05 * x ** 2
          + np.sin(y) + 0.05 * y ** 2)


# In[55]:

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
Z = fm((X, Y))


# In[56]:

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm,
        linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
# tag: opt_plot_3d
# title: Function to minimize with two parameters
# size: 60


# In[57]:

import scipy.optimize as spo


# ### Global Optimization

# In[58]:

def fo((x, y)):
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output == True:
        print '%8.4f %8.4f %8.4f' % (x, y, z)
    return z


# In[59]:

output = True
spo.brute(fo, ((-10, 10.1, 5), (-10, 10.1, 5)), finish=None)


# In[60]:

output = False
opt1 = spo.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish=None)
opt1


# In[61]:

fm(opt1)


# ### Local Optimization

# In[62]:

output = True
opt2 = spo.fmin(fo, opt1, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
opt2


# In[63]:

fm(opt2)


# In[64]:

output = False
spo.fmin(fo, (2.0, 2.0), maxiter=250)


# ### Constrained Optimization

# In[65]:

# function to be minimized
from math import sqrt
def Eu((s, b)):
    return -(0.5 * sqrt(s * 15 + b * 5) + 0.5 * sqrt(s * 5 + b * 12))

# constraints
cons = ({'type': 'ineq', 'fun': lambda (s, b):  100 - s * 10 - b * 10})
  # budget constraint
bnds = ((0, 1000), (0, 1000))  # uppper bounds large enough


# In[66]:

result = spo.minimize(Eu, [5, 5], method='SLSQP',
                       bounds=bnds, constraints=cons)


# In[67]:

result


# In[68]:

result['x']


# In[69]:

-result['fun']


# In[70]:

np.dot(result['x'], [10, 10])


# ## Integration

# In[71]:

import scipy.integrate as sci


# In[72]:

def f(x):
    return np.sin(x) + 0.5 * x


# In[73]:

a = 0.5  # left integral limit
b = 9.5  # right integral limit
x = np.linspace(0, 10)
y = f(x)


# In[74]:

from matplotlib.patches import Polygon

fig, ax = plt.subplots(figsize=(7, 5))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)

# area under the function
# between lower and upper limit
Ix = np.linspace(a, b)
Iy = f(Ix)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)

# labels
plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$",
         horizontalalignment='center', fontsize=20)

plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')

ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([f(a), f(b)])
# tag: sin_integral
# title: Example function with integral area
# size: 50


# ### Numerical Integration

# In[75]:

sci.fixed_quad(f, a, b)[0]


# In[76]:

sci.quad(f, a, b)[0]


# In[77]:

sci.romberg(f, a, b)


# In[78]:

xi = np.linspace(0.5, 9.5, 25)


# In[79]:

sci.trapz(f(xi), xi)


# In[80]:

sci.simps(f(xi), xi)


# ### Integration by Simulation

# In[81]:

for i in range(1, 20):
    np.random.seed(1000)
    x = np.random.random(i * 10) * (b - a) + a
    print np.sum(f(x)) / len(x) * (b - a)


# ## Symbolic Computation

# In[82]:

import sympy as sy


# ### Basics

# In[83]:

x = sy.Symbol('x')
y = sy.Symbol('y')


# In[84]:

type(x)


# In[85]:

sy.sqrt(x)


# In[86]:

3 + sy.sqrt(x) - 4 ** 2


# In[87]:

f = x ** 2 + 3 + 0.5 * x ** 2 + 3 / 2


# In[88]:

sy.simplify(f)


# In[89]:

sy.init_printing(pretty_print=False, use_unicode=False)


# In[90]:

print sy.pretty(f)


# In[91]:

print sy.pretty(sy.sqrt(x) + 0.5)


# In[92]:

pi_str = str(sy.N(sy.pi, 400000))
pi_str[:40]


# In[93]:

pi_str[-40:]


# In[94]:

pi_str.find('111272')


# ### Equations

# In[95]:

sy.solve(x ** 2 - 1)


# In[96]:

sy.solve(x ** 2 - 1 - 3)


# In[97]:

sy.solve(x ** 3 + 0.5 * x ** 2 - 1)


# In[98]:

sy.solve(x ** 2 + y ** 2)


# ### Integration

# In[99]:

a, b = sy.symbols('a b')


# In[100]:

print sy.pretty(sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b)))


# In[101]:

int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)


# In[102]:

print sy.pretty(int_func)


# In[103]:

Fb = int_func.subs(x, 9.5).evalf()
Fa = int_func.subs(x, 0.5).evalf()


# In[104]:

Fb - Fa  # exact value of integral


# In[105]:

int_func_limits = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b))
print sy.pretty(int_func_limits)


# In[106]:

int_func_limits.subs({a : 0.5, b : 9.5}).evalf()


# In[107]:

sy.integrate(sy.sin(x) + 0.5 * x, (x, 0.5, 9.5))


# ### Differentiation

# In[108]:

int_func.diff()


# In[109]:

f = (sy.sin(x) + 0.05 * x ** 2
   + sy.sin(y) + 0.05 * y ** 2)


# In[110]:

del_x = sy.diff(f, x)
del_x


# In[111]:

del_y = sy.diff(f, y)
del_y


# In[112]:

xo = sy.nsolve(del_x, -1.5)
xo


# In[113]:

yo = sy.nsolve(del_y, -1.5)
yo


# In[114]:

f.subs({x : xo, y : yo}).evalf() 
  # global minimum


# In[115]:

xo = sy.nsolve(del_x, 1.5)
xo


# In[116]:

yo = sy.nsolve(del_y, 1.5)
yo


# In[117]:

f.subs({x : xo, y : yo}).evalf()
  # local minimum


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
