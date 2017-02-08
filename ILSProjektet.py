import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, criterion , max_features, max_depth , min_samples_leaf):

        self.criterion= criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = Node(None)

    def fitness(self, X,):
        self.root.insert(X)

    def str_column_to_float(dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())




class Node:

    def __init__(self, parent):
        self.parent = parent
        self.left = None
        self.right = None
        self.values = None
        self.range = None
        self.indexSplit = None
        self.ValueSplit = None
        self.list = np.array

    def insert(self, X):

        self.range = X[0:1].size -1
        self.values = X
        self.split(X)

    def split(self, X):

        for i in range(self.range):
            test = []
            for j in range(len(X)):
                value = X[j, i]
                if not(value in test):
                    test.append(value)
            print(test)
        print(self.list)



def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())



data = pd.read_csv(r"F:\Tree\binary\balance-scale.csv", header = None)

npdata = np.array(data[1:])
for i in range(len(npdata[0])-1):
	str_column_to_float(npdata[0:, 0:4], i)
print(npdata)


#print(npdata)

tree = DecisionTree(1,1,1,1)
tree.fitness(npdata)