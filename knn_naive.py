import pandas as pd
import numpy as np
import math
import operator
from scipy.stats import mode

df = pd.read_csv('car_data.csv')
df

K = int(input('Enter K : '))

def replaceData(dataframe):
    columns = dataframe.columns.values.tolist()

    for i in range(len(columns) - 1):
        replacementList = list(range(1, len(dataframe[columns[i]].unique()) + 1))
        dataframe[[columns[i]]] = dataframe[[columns[i]]].replace(dataframe[columns[i]].unique(), replacementList)
    return dataframe


def train_test_split(X, Y, test_size=0.25):
    length = X.shape[0]
    indeces = list(range(0, length))
    np.random.shuffle(indeces)

    number_of_rows = int(length * test_size)
    trainIndeces = indeces[number_of_rows:]
    testIndeces = indeces[:number_of_rows]

    X_train = X.iloc[trainIndeces]
    X_test = X.iloc[testIndeces]

    Y_train = Y.iloc[trainIndeces]
    Y_test = Y.iloc[testIndeces]

    return X_train, X_test, Y_train, Y_test


def EuclideanDistance(firstPoint, secondPoint):
    sum = 0.0
    if len(firstPoint) == len(secondPoint):
        for i in range(len(firstPoint)):
            sum += math.pow(firstPoint[i] - secondPoint[i], 2)
    return math.sqrt(sum)


def ManhattanDistance(firstPoint, secondPoint):
    sum = 0.0
    if len(firstPoint) == len(secondPoint):
        for i in range(len(firstPoint)):
            sum += abs(firstPoint[i] - secondPoint[i])
    return sum


def DistanceCalc(type, firstPoint, SecondPoint):
    if type == 'Euclidean':
        return EuclideanDistance(firstPoint, SecondPoint)
    elif type == 'Manhattan':
        return ManhattanDistance(firstPoint, SecondPoint)


def KNN_predict(X_train, X_test, Y_train, K, dType):
    Y_pred = []

    X_train_values = X_train.values.tolist()
    X_test_values = X_test.values.tolist()
    Y_train_values = Y_train.values.tolist()

    for i in range(X_test.shape[0]):
        distances = []
        neighbours = []
        for j in range(X_train.shape[0]):
            distances.append(DistanceCalc(dType, X_test_values[i], X_train_values[j]))
        npDistances = np.array(distances)
        neighboursIndeces = np.argsort(npDistances)[:K]

        classifications = []
        for c in range(K):
            classifications.append(Y_train_values[neighboursIndeces[c]])
        classification = mode(classifications)
        classification = classification.mode[0]

        Y_pred.append(classification)

    return Y_pred


def KNN_accuracy(Y_pred, Y_test):
    return np.mean(Y_pred == Y_test)


df2 = df.copy()
df2 = replaceData(df2)
X_train, X_test, Y_train, Y_test = train_test_split(df2.iloc[:, :-1], df2.iloc[:, -1])
Y_pred = KNN_predict(X_train, X_test, Y_train, K, 'Euclidean')
Y_pred

accuracy = KNN_accuracy(Y_pred, Y_test)
print('Accuracy of KNN: ', accuracy)


def calculate_probabilities(features, target):
    probabilities = {}
    target_column_name = target.name
    classes = target.unique()

    for column in features.columns.values.tolist():
        subFrame = features[[column]].copy()
        unique_values = features[column].unique()

        all_cols = pd.concat([subFrame, target], axis=1)
        count_series = all_cols.groupby([column, target_column_name]).size()
        combinations = count_series.to_frame(name='join_count').reset_index()

        for value in unique_values:
            for target_class in classes:
                row = combinations[(combinations[column] == value) & (combinations[target_column_name] == target_class)]
                if row.empty:
                    join_size = 0
                else:
                    join_size = int(row.join_count)
                join_prob = join_size / target[target == target_class].shape[0]

                probabilities[(column, target_class, value)] = join_prob

    return probabilities


def Naive_Bayes(X_train, X_test, Y_train):
    probabilities = calculate_probabilities(X_train, Y_train)
    class_final_probabilities = {}
    Y_pred = []
    target_probs = {}

    X_train_values = X_train.values.tolist()
    X_test_values = X_test.values.tolist()
    columns = X_train.columns.tolist()
    classes = Y_train.unique()

    for target2 in classes:
        target_probs[target2] = Y_train[Y_train == target2].shape[0] / Y_train.shape[0]

    for row in X_test_values:
        probs = {}
        for target in classes:
            probs[target] = target_probs[target]

        for i in range(len(row)):
            for target_class in classes:
                probs[target_class] = probs[target_class] * probabilities[(columns[i], target_class, row[i])]

        result = max(probs.items(), key=operator.itemgetter(1))[0]
        Y_pred.append(result)

    return Y_pred


def Naive_accruacy(Y_pred, Y_test):
    return np.mean(Y_pred == Y_test)


df3 = df.copy()
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(df3.iloc[:, :-1], df3.iloc[:, -1])
Y_pred2 = Naive_Bayes(X_train2, X_test2, Y_train2)
Y_pred2

accuracy2 = Naive_accruacy(Y_pred2, Y_test2)
print('Accuracy of Naive Bayes : ', accuracy2)
