# coding: utf-8

#import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv


#=================================================
# csvファイルからデータを取り出し、listに格納
def set_data():

	filename = "../data/misyo/data_standardized"
	
	f = open('%s.csv' % filename, 'rU')
	data = csv.reader(f)
	
	data_set = []
	target_set = []
	for line in data:
		data_set.append(line[1:4])
		target_set.append(line[4])
	f.close()
	np_dataSet = np.array(data_set, dtype=np.float32)
	np_targetSet = np.array(target_set, dtype=np.int32)
	return np_dataSet, np_targetSet

#=================================================

data, target = set_data()

# <!--- start_debug
#print data
#print target
#       end_debug ----> 

Brix_upper = data[:,0]
Brix_middle = data[:,1]
Brix_average = data[:,2]

plt.hist( Brix_upper, bins = 20)
plt.xlabel('Brix_upper_standardized')
plt.ylabel('freaquency')
plt.title('Histgram')
plt.show()
