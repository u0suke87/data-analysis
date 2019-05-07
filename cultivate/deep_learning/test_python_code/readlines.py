# conding: UTF-8

f = open('data.csv')
lines2 = f.readlines()
f.close()

for line in lines2:
	print line,
print
