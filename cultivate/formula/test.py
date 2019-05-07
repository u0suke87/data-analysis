#coding: utf-8:
import numpy as np
import matplotlib.pyplot as plt

#constant 
a = 2.5
b = 6

x = np.arange(-4, 4, 0.025)
exp = np.exp( -a * x)

function = ( b / ( 1 + exp) ) - (b / 2) 

plt.plot(x, function, lw=2, alpha=0.7, label="function")

plt.xlabel("$x$", fontsize=20)
plt.ylabel("$y$", fontsize=20)
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.grid()
plt.show()

