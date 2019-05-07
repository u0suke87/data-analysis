import numpy as np

def a():
	return np.array[1,2]

def b():
	return 3,4

def c():
	f,g = a()
	h,i = b()
	return f,g,h,i

def j():
	return a(), 1
def k():
	return 5
def l():
	return 6
m, n = j()
print m,n
#d,e = a()
#print d,e
#
#j,k,l,m = c()
#print j,k,l,m
#d,e,f,g = c()
#print d,e,f,g
