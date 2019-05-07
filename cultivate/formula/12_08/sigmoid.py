#coding: utf-8:
import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

#constant 

x_data = np.linspace (-10, 10, 100, dtype=np.float32)
x = Variable(x_data)

y = F.sigmoid(x)

plt.plot(x.data, y.data)

plt.title( "Sigmoid	' f(x) = ( 1 + exp(-x) )^(-1) '  ")
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$f(x)$", fontsize=20)
plt.xlim([-4,4])
plt.ylim([-0.5,1.5])
plt.grid()
plt.show()
