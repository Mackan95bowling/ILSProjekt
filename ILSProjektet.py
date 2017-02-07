import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

data = pd.read_csv(r"F:\Tree\binary\balance-scale.csv", header = None)

npdata = np.array(data)
size = npdata[0:1].size
X = npdata[1:-1, 0:size-1]
y = npdata[1:-1, size-1]
##print(y)
print(X)
##y = np.sin(test).ravel()
##print(test)
# Fit regression model




class DecisionTree:

   def __init__(self,
                  criterion ,
                  max_features,
                  max_depth ,
                  min_samples_leaf):

      self.criterion= criterion
      self.max_features = max_features
      self.max_depth = max_depth
      self.min_samples_leaf = min_samples_leaf