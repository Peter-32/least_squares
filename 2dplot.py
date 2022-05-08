import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
rcParams['figure.figsize'] = 10, 8

x = np.random.uniform(0, 10, size=100)
y = np.array((2.1 * x) + [random.gauss(0,2) for x in range(100)])
lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
pred_x = np.linspace(0,10).reshape(-1, 1)
pred_y = lr.predict(pred_x)
plt.plot(pred_x, pred_y, color='lightblue')
plt.scatter(x, y, color='gray')
plt.show()
