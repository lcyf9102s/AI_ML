import numpy as np
from random import *
import matplotlib.pyplot as plt

# 生成直线周围的一组数据集
x_list = 20 * np.random.rand(500)
x_list.shape[0]

def x_to_yList(xlist, f):
    return [f(x) for x in xlist]

y_list = x_to_yList(x_list, lambda x: 2.44 * x + (5.36 + gauss(0, 3)))
m = x_list.shape[0]

# 模型构造
def h(t0, t1, x):
    r = t0 + t1 * x
    return r

# 构建代价函数
def cost_f(t0, t1, x1, y1):
    delta = t0 + t1 * x1 - y1
    cost = np.inner(delta, delta) / (2 * m)
    return cost

# 代价函数计算偏导
def de(t0, t1, _x, _y):
    delta = t0 + t1 * _x - _y
    de0 = np.inner(delta, np.ones(m)) / m
    de1 = np.inner(x_list, delta) / m
    return de0, de1

# 使用梯度下降法迭代得到最佳参数
def grad_decent(_theta0, _theta1, x, y, epoch, alpha):
    for i in range(epoch):
        cost1 = cost_f(t0 = _theta0, t1 = _theta1, x1 = x, y1 = y)
        de0, de1 = de(t0 = _theta0, t1 = _theta1, _x = x, _y = y)
        _theta0 = _theta0 - alpha * de0
        _theta1 = _theta1 - alpha * de1
        cost2 = cost_f(t0 = _theta0, t1 = _theta1, x1 = x, y1 = y)
        if cost1 - cost2 < 1e-10:
            break
    return _theta0, _theta1

theta0 = 0.0
theta1 = 0.0
theta0, theta1 = grad_decent(theta0, theta1, x = x_list, y = y_list, epoch = 5000000, alpha = 0.0001)

plt.plot(x_list, (theta0 + theta1 * x_list), color = 'yellow')
plt.scatter(x_list, y_list)
plt.show()