# -*- coding: utf-8 -*-
"""K-Means.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oJ4-x4zXmt873SfVI-ixA2BCojGsTPEl
"""

import pandas as pd
import numpy as np
import math
import random

K = int(input('Enter K : '))
distance_type = input('Enter Distance Type : ')
normalization_type = input('Enter Normalization Type : ')

df = pd.read_excel('Odor_Meter_Results.xlsx', sheet_name='Odor_Meter_Results', header=None,engine='openpyxl')
df2 = df.copy()

df.drop(columns=df.columns[0], axis=1, inplace=True)
df_numbers = df
df_names = df2.iloc[:, :1]


def StandardNormalization(dataframe):
    return ((dataframe - dataframe.mean()) / dataframe.std())


def MinmaxNormalization(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())


def Normalize(dataframe, type):
    if type == 'Zscore':
        return StandardNormalization(dataframe)
    elif type == 'MinMax':
        return MinmaxNormalization(dataframe)


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


def centroidCalc(cluster):
    centroid = []
    for i in range(len(cluster[0])):
        sum = 0.0
        for j in range(len(cluster)):
            sum += cluster[j][i]
        centroid.append(sum / len(cluster))
    return centroid


def outlierDetector(distances):
    outliers = []
    distances_sorted = np.array(distances)
    sorted(distances_sorted)
    quantile1, quantile3 = np.percentile(distances_sorted, [25, 75])
    iqr = quantile3 - quantile1
    upperBound = quantile3 + (1.5 * iqr)
    for distance in distances:
        if distance > upperBound:
            outliers.append(True)
        else:
            outliers.append(False)
    return outliers


def KMeans(K, dfnumbers, dfnames, dType, nType):
    dfnumbers = Normalize(dfnumbers, nType)
    numbers = dfnumbers.values.tolist()
    names = dfnames.values.tolist()

    K_means_output = {}
    centroids = []
    indeces = random.sample(range(0, len(numbers)), K)
    for i in indeces:
        centroids.append(numbers[i])

    prevCentroids = []

    while prevCentroids != centroids:
        clusters = []
        for point in numbers:
            distances = []
            for centroid in centroids:
                distances.append(DistanceCalc(dType, point, centroid))
            num_of_cluster = distances.index(min(distances)) + 1
            clusters.append(num_of_cluster)
        prevCentroids = centroids.copy()
        for i in range(K):
            cluster = [numbers[x] for x in range(len(numbers)) if clusters[x] == i + 1]
            centroids[i] = centroidCalc(cluster)

    all_Distances = []
    for i in range(len(numbers)):
        all_Distances.append(DistanceCalc(dType, numbers[i], centroids[clusters[i] - 1]))

    outliers = outlierDetector(all_Distances)

    finalised_clusters = []

    for i in range(len(numbers)):
        finalised_clusters.append([])
        finalised_clusters[i].append(names[i][0])
        finalised_clusters[i].extend(numbers[i])

    for i in range(K):
        cluster = []
        outlierCluster = []
        for x in range(len(finalised_clusters)):
            if clusters[x] == i + 1:
                if outliers[x]:
                    outlierCluster.append(finalised_clusters[x])
                else:
                    cluster.append(finalised_clusters[x])
        K_means_output['Cluster' + str(i + 1)] = cluster
        K_means_output['Cluster' + str(i + 1) + ' Outliers'] = outlierCluster

    return K_means_output , all_Distances


KmeansOutput , distances = KMeans(K, df_numbers, df_names, distance_type, normalization_type)

for key in KmeansOutput.items():
    print(key[0], end=": \n")
    for i in range(len(key[1])):
        print(key[1][i][0], end="\n")
    print(end="\n")

namesList = df_names.values.tolist()
for a in range(len(namesList)):
    print(namesList[a], '\t', distances[a])
