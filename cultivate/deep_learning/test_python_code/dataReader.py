#coding:UTF-8

import csv
import numpy as np

#=================================================
def set_data():

	f = open('data.csv', 'rU')
	
	dataReader = csv.reader(f)
		
#	for row in dataReader:
#		print row[1]

	data_set = []
	target_set = []
	for data in dataReader:
		data_set.append(data[:4])
		target_set.append(data[4])
	f.close()
	np_dataSet = np.array(data_set, dtype=np.float32)
	np_targetSet = np.array(target_set, dtype=np.int32)
	return np_dataSet, np_targetSet
#	return data_set, target_set

#=================================================

#data, target = np.array(set_data())
data, target = set_data()


print target
print 
print data
