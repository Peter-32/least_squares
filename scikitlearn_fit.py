import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
import random
rcParams['figure.figsize'] = 10, 8

random.seed(10)
np.random.seed(10)
lr = LinearRegression()

pred_x = pd.DataFrame(np.linspace(0,10, num=100))
pred_y = pd.DataFrame(np.linspace(0,5, num=100))
pred_x.columns = ['x1']
pred_y.columns = ['x2']
pred_x['key'] = 1
pred_y['key'] = 1
X = pred_x.join(pred_y.set_index('key'), on='key', how='inner')
X.drop(['key'], axis='columns', inplace=True)
X['x0'] = 1
pred_x = X['x1']
pred_y = X['x2']
X = X[['x0', 'x1', 'x2']]

def get_true_z(x1, x2):
    return (2.1*x1) + (2.5*x2) + random.gauss(0,2)

true_z = X[['x1', 'x2']].apply(lambda x: get_true_z(*x), axis='columns')
X['true_z'] = true_z

X.drop(['true_z', 'x0'], axis='columns', inplace=True)

y = pd.DataFrame(true_z)
y.columns = [0]

model = lr.fit(X, y)
print(model.intercept_, model.coef_)
