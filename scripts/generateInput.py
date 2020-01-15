#!/usr/bin/python

import sys
import random

random.seed(0)

n = int(sys.argv[1].strip())
m = int(sys.argv[2].strip())

filename = str(sys.argv[3].strip())
file = open(filename, 'w')

file.write(str(n) + '\n')
file.write(str(m) + '\n')

for i in range(n):
	for j in range(m):
		x = random.uniform(0.0, 10.0)
		file.write(str(x))
		if j < m-1:
			file.write(',')
	file.write('\n')

file.close()
