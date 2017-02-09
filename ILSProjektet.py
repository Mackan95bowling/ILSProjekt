import numpy as np
import pandas as pd
import sklearn.cross_validation



class DecisionTree:
    def __init__(self, criterion , max_features, max_depth , min_samples_leaf):

        self.criterion= criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = Node(None, max_depth, min_samples_leaf)

    def fitness(self, X,):
        self.root.insert(X)

    def predict(self, X):
        predictions = []
        for item in X:
            prediction = self.root.search(item)
            predictions.append(prediction)
        return(predictions)


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

    def search(self, X):

        if self.classValue == None:
            if X[self.indexSplit] > self.ValueSplit:
                predict = self.left.search(X)
            else:
                predict = self.right.search(X)
            return(predict)
        else:
            return(self.classValue)


    def insert(self, X):
        self.range = X[0:1].size -1
        self.max_depth = self.max_depth -1
        if (self.allDistingt(X)):
            self.classValue= self.findClassValue(X)

        elif (self.max_depth ==0):
            self.classValue= self.findClassValue(X)

        elif (len(X)< self.min_sample_leaf):
            self.classValue= self.findClassValue(X)

        else:
            self.values = X
            self.findValues(X)
            self.testSplit(X)
            self.split(X)

    def allDistingt(self, X):
        distClassValues = []
        for item in X:
            if not(distClassValues.__contains__(item[self.range])): # save distingt classes
                distClassValues.append(item[self.range])
                if (len(distClassValues) > 1):
                    return(False)
        return(True)

    def findClassValue(self, X):
        distClassValues = []
        for item in X:
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
        self.left = Node(self, self.max_depth, self.min_sample_leaf)
        self.right = Node(self, self.max_depth, self.min_sample_leaf)
        self.left.insert(numpyLeft)
        self.right.insert(numpyRight)


    def findValues(self, X):
        self.list = []
        for i in range(self.range):
            test = []
            for j in range(X.size/(self.range+1)):
                value = X[j, i]
                if not(test.__contains__(value)):
                    test.append(value)
            self.list.append(test)


    def gini(self, left, right):
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
        return(giniChildren)



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

                giniValue = self.gini(left, right)
                if giniValue < bestGini:
                    bestGini = giniValue
                    self.indexSplit = i
                    self.ValueSplit = self.list[i][j]

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def refactor_data(X,Y):
    row = []
    i = 0
    for element in X:
        out = np.append(element, Y[i])
        row.append(out)
        i = i + 1
    data = np.array(row)
    for i in range(len(data[0])-1):
        str_column_to_float(data[0:, :-1], i)
    return(data)

def getAccuracy(prediction, answer):
    total = 0
    for i in range(len(prediction)):
        if prediction[i] == answer[i]:
            total = total + 1
    accuracy = total / (float)(i + 1)
    return(accuracy)







data = pd.read_csv(r"F:\Tree\binary\balance-scale.csv", header = None)
npdata = np.array(data)
X_data = npdata[1:, :-1]
y_data = npdata[1:, -1:]


X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_data, y_data, test_size=0.33, random_state=42)


my_trainData = refactor_data(X_train, y_train)
my_predictData = refactor_data(X_test, y_test)



tree = DecisionTree(1,1,20,30)
tree.fitness(my_trainData)
prediction = tree.predict(my_predictData)
accuracy = getAccuracy(prediction, y_test)
print(accuracy)