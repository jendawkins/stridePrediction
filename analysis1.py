
import glob
import pdb
import numpy as np
import csv
import matplotlib.pyplot as plt
%matplotlib inline

x = dict()
for file_name in glob.glob('*.csv'):
    str2 = str.replace(file_name, '.','_')
    str3 = str.split(str2, '_')
    x['_'.join(str3[2:4])] = {}
    with open(file_name, 'r') as infile:
        data = infile.read().splitlines()
        data.pop(0)
        data.pop(-1)
        for i, dataline in enumerate(data):
            data[i] = dataline.split(',')[1:]
        header = data.pop(0)
        data = np.array(data)
        for i,h in enumerate(header):
            x['_'.join(str3[2:4])][h] = data[:,i]

        plt.figure()
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(data[:,0], data[:,2])
        plt.scatter(data[:,0], data[:,3])
        plt.title('Acceleration: ' + ', '.join(str3[2:4]))
        plt.show()

        plt.figure()
        plt.scatter(data[:,0], data[:,4])
        plt.scatter(data[:,0], data[:,5])
        plt.scatter(data[:,0], data[:,6])
        plt.title('Gyro: ' + ', '.join(str3[2:4]))
        plt.show()
