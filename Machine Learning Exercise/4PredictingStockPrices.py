import csv
import numpy as numpy
from sklearn.svm import SVR
import matplotlib.pyplot as plt 

dates = []
prices = []

def get_data (filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			