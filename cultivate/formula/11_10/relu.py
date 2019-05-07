#coding: utf-8:
import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

#constant 

x_data = np.linspace (-10, 10, 100, dtype=np.float32)
x = Variable(x_data)

y = F.relu(x)

plt.plot(x.data, y.data)

plt.title( "ReLu	' f(x) = max(0, x)'  ")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$f(x)$", fontsize=20)
plt.xlim([-4,4])
plt.ylim([-2,6])
plt.grid()
plt.show()

