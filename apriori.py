import pandas as pd
import numpy as np
import math
from itertools import combinations

min_support = float(input("Enter Minimum Support :"))

min_confidence_rule = float(input("Enter Minimum Confidence : "))

df = pd.read_csv(r"F:\New folder (2)\DataMining_Assignment1\retail_dataset.csv")
df.drop(columns=df.columns[0], axis=1, inplace=True)
df.fillna('NaN', inplace=True)
df_transactions = df.values.tolist()
for i in range(len(df_transactions)):
    df_transactions[i] = [x for x in df_transactions[i] if x != 'NaN']

min_support_count = int(min_support * len(df_transactions))


def filtering(dataframe, minSupport):
    return dataframe[dataframe.SupportCount >= minSupport]


def count_Item(transactions):
    itemsDict = {}
    for transaction in transactions:
        for i in range(len(transaction)):
            if transaction[i] in itemsDict.keys():
                itemsDict[transaction[i]] = itemsDict[transaction[i]] + 1
            else:
                itemsDict[transaction[i]] = 1

    dataframe = pd.DataFrame()
    dataframe['ItemSets'] = itemsDict.keys()
    dataframe['SupportCount'] = itemsDict.values()
    dataframe = dataframe.sort_values('ItemSets')
    return dataframe


def count_Itemsets(transactions, itemsets):
    itemsDict = {}
    for itemset in itemsets:
        set1 = set(itemset)
        for transaction in transactions:
            transaction_set = set(transaction)
            if transaction_set.intersection(set1) == set1:
                if itemset in itemsDict.keys():
                    itemsDict[itemset] = itemsDict[itemset] + 1
                else:
                    itemsDict[itemset] = 1

    dataframe = pd.DataFrame()
    dataframe['ItemSets'] = itemsDict.keys()
    dataframe['SupportCount'] = itemsDict.values()
    return dataframe


def join(items):
    itemsets = []
    i = 1
    for Item in items:
        rest_of_items = items[i:]
        for item2 in rest_of_items:
            if (type(item2) is str):
                if Item != item2:
                    tupleSet = (Item, item2)
                    itemsets.append(tupleSet)
            else:
                if Item[0:-1] == item2[0:-1]:
                    tupleSet = Item + item2[1:]
                    tupleSet = tuple(set(tupleSet))
                    itemsets.append(tupleSet)
        i = i + 1
    if (len(itemsets) == 0):
        return None
    return itemsets


def Apriori(transactions, minSupport):
    frequents = pd.DataFrame()
    dataframe = count_Item(transactions)

    while (len(dataframe) != 0):
        dataframe = filtering(dataframe, minSupport)

        if len(dataframe) > 1 or (len(dataframe) == 1 and int(dataframe.SupportCount >= minSupport)):
            frequents = dataframe

        itemsets = join(dataframe.ItemSets)

        if (itemsets is None):
            return frequents

        dataframe = count_Itemsets(transactions, itemsets)

        #print(dataframe)

    return dataframe


frequentItemsets = Apriori(df_transactions, min_support_count)
print(frequentItemsets)


def calculateConfidence(support1, support2):
    return round(int(support1) / int(support2), 2)


def ConfidenceRule(frequent_Item_Sets, minConfidence):
    confidences = {}
    if frequent_Item_Sets.empty:
        return
    for item_set in frequent_Item_Sets.ItemSets:
        if type(item_set) is str:
                key = item_set
                confidences[key] = 1
        else :
            set1 = set(item_set)
            for base_length in range(len(item_set)):
                for combination_set in combinations(item_set, base_length):
                    items_base = combination_set
                    items_add = set1.difference(items_base)
                    key = str(items_base) + '🠮' + str(items_add)
                    if len(items_base) > 1:
                        listoflists = [items_base]
                        confidence = calculateConfidence(
                            frequent_Item_Sets[frequent_Item_Sets.ItemSets == item_set].SupportCount,
                            count_Itemsets(df_transactions, listoflists).SupportCount)
                        confidences[key] = confidence
                    elif len(items_base) == 1:
                        confidence = calculateConfidence(
                            frequent_Item_Sets[frequent_Item_Sets.ItemSets == item_set].SupportCount,
                            count_Item(df_transactions)[count_Item(df_transactions).ItemSets == items_base[0]].SupportCount)
                        confidences[key] = confidence

    dataframe = pd.DataFrame()
    dataframe['ItemSets'] = confidences.keys()
    dataframe['Confidence'] = confidences.values()
    return dataframe[dataframe.Confidence >= minConfidence]


confidenceItemSets = ConfidenceRule(frequentItemsets, min_confidence_rule)
print(confidenceItemSets)

