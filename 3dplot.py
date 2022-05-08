import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
rcParams['figure.figsize'] = 10, 8

lr = LinearRegression()

pred_x = pd.DataFrame(np.linspace(0,10, num=100))
pred_y = pd.DataFrame(np.linspace(0,5, num=100))
pred_x.columns = ['x1']
pred_y.columns = ['x2']
pred_x['key'] = 1
pred_y['key'] = 1
X = pred_x.join(pred_y.set_index('key'), on='key', how='inner')
X.drop(['key'], axis='columns', inplace=True)
pred_x = X['x1']
pred_y = X['x2']

def get_true_z(x1, x2):
    return (2.1*x1) + (2.5*x2) + random.gauss(0,2)

true_z = X[['x1', 'x2']].apply(lambda x: get_true_z(*x), axis='columns')
X['true_z'] = true_z
X_sample = X.copy()
X_sample = X_sample.sample(35)
x = X_sample['x1']
y = X_sample['x2']
z = X_sample['true_z']

X.drop(['true_z'], axis='columns', inplace=True)
lr.fit(X, true_z)
pred_z = lr.predict(X)
fig = plt.figure()
ax = plt.axes(projection= '3d')
ax.scatter(x, y, z, color='gray', alpha=0.8)
ax.plot(pred_x, pred_y, pred_z, color='lightblue', alpha=0.7)

plt.show()
