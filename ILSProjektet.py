import numpy as np
import pandas as pd





class DecisionTree:
    def __init__(self, criterion , max_features, max_depth , min_samples_leaf):

        self.criterion= criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = Node(None, max_depth, min_samples_leaf)

    def fitness(self, X,):
        self.root.insert(X)

class Node:

    def __init__(self, parent, depth, min_sample_leaf):
        self.parent = parent
        self.max_depth = depth
        self.min_sample_leaf = min_sample_leaf
        self.left = None
        self.right = None
        self.values = None
        self.range = None
        self.indexSplit = None
        self.ValueSplit = None
        self.list = None
        self.classValue = None



    def insert(self, X):
        self.range = X[0:1].size -1
        self.max_depth = self.max_depth -1
        if (self.max_depth ==0):
            self.classValue= self.findClassValue(X)


        elif (len(X)< self.min_sample_leaf):
            self.classValue= self.findClassValue(X)


        else:

            self.values = X
            self.findValues(X)
            self.testSplit(X)
            self.split(X)

    def findClassValue(self, X):
        distClassValues = []
        for item in X:
            print(item[self.range])
            if not(distClassValues.__contains__(item[self.range])): # save distingt classes
                distClassValues.append(item[self.range])

        count = np.zeros(len(distClassValues))
        for j in range(len(distClassValues)):
            for item in X:
                if distClassValues[j] == item[self.range]:
                    count[j] = count[j] + 1
        max = 0
        for i in range(len(count)):
            if count[i] > max:
                max = count[i]
                classValue = distClassValues[i]
        print(classValue)
        return(classValue)

    def split(self, X):
        left = []
        right = []
        for row in X:
            if row[self.indexSplit] > self.ValueSplit:
                left.append(row)
            else:
                right.append(row)
        numpyLeft = np.array(left)
        numpyRight= np.array(right)


    def findValues(self, X):
        self.list = []
        for i in range(self.range):
            test = []
            for j in range(X.size/(self.range+1)):
                value = X[j, i]
                if not(test.__contains__(value)):
                    test.append(value)
            self.list.append(test)


    def gini(self, child):
        distinctClasses = []
        for item in child:
            if not(distinctClasses.__contains__(item)): # save distingt classes
                distinctClasses.append(item)

        count = np.zeros(len(distinctClasses))
        for j in range(len(distinctClasses)):
            for i in range(len(child)):
                if distinctClasses[j] == child[i]:
                    count[j] = count[j] + 1
        totalLeft = len(child)

        giniValue = 1
        for value in count:
            giniValue = giniValue - (value/totalLeft)**2

        return(giniValue)



    def testSplit(self, X):
        bestGini = 999
        for i in xrange(self.range):
            for j in xrange(len(self.list[i])-1):
                left = []
                right = []
                for row in X:
                    lastIndex = len(row)-1
                    if row[i] > self.list[i][j]:
                        left.append(row[lastIndex])
                    else:
                        right.append(row[lastIndex])
                giniLeft = self.gini(left)
                giniRight = self.gini(right)


                """giniValue = self.gini(left, right)
                if giniValue < bestGini:
                    bestGini = giniValue
                    self.indexSplit = i
                    self.ValueSplit = self.list[i][j]"""





def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

data = pd.read_csv(r"F:\Tree\binary\balance-scale.csv", header = None)
npdata = np.array(data[1:])
tree = DecisionTree(1,1,10,800)
for i in range(len(npdata[0])-1):
    str_column_to_float(npdata[0:, 0:4], i)
tree.fitness(npdata)








"""    def gini(self, left, right):
        leftClass = []
        for item in left:
            if not(leftClass.__contains__(item)): # save distingt classes in left
                leftClass.append(item)
        rightClass = []
        for item in right:
            if not(rightClass.__contains__(item)): # save distingt classes in right
                rightClass.append(item)

        countLeft = np.zeros(len(leftClass))
        for j in range(len(leftClass)):
            for i in range(len(left)):
                if leftClass[j] == left[i]:
                    countLeft[j] = countLeft[j] + 1
        totalLeft = len(left)

        countRight = np.zeros(len(rightClass))
        for k in range(len(rightClass)):
            for l in range(len(right)):
                if rightClass[k] == right[l]:
                    countRight[k] = countRight[k] + 1
        totalRight = len(right)

        giniLeft  = 1
        for value in countLeft:
            giniLeft = giniLeft - (value/totalLeft)**2

        giniRight = 1
        for value in countRight:
            giniRight = giniRight - (value/totalRight)**2

        childrenTotal = float(totalRight) + float(totalLeft)
        giniChildren = (((totalLeft/childrenTotal) * giniLeft) + ((totalRight/childrenTotal) * giniRight))
        allGini = [giniLeft, giniRight, giniChildren]
        print(allGini)
        return(giniChildren)"""