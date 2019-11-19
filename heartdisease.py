import pandas as pd
import math
from collections import Counter
from sklearn.utils import shuffle

def euclidianDistance(a, b):
    # Gets the euclidean distance from 2 n-dimensional points
    # (n=13 in this case)

    A = a[2:14]
    B = b[2:14]
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))

def buildDistanceMatrix(dataset, unknown):
    # Array of tuples
    # Where entry is: (distance, indexInDataset)
    # And distance is the distance from a point in the dataset to the unknown

    distanceIndex = []
    for i in range(len(dataset)):
        dst = euclidianDistance(dataset[i], unknown)
        distanceIndex.append((dst, i))
    
    return distanceIndex

def getNearest(k, dstindex):
    # Gets the K nearest neighbours from the distance matrix
    return sorted(dstindex, key=(lambda x: x[0]))[:k]

def getValue(closest, ds):
    # Gets the predicted value from the K nearest neighbours
    
    lst = []
    for p in closest:
        lst.append(ds[p[1]][13])
    oc = Counter(lst)
    return oc.most_common(1)[0][0]

def acc(pred, test):
    # Gets accuracy 
    l = len(test)
    t = 0
    for i in range(len(pred)):
        if pred[i] == test[i][13]:
            t += 1
    return t/l

def normalize(ds):
    # Normalize columns 0, 3, 4, and 7

    v0 = [t[0] for t in ds]
    v3 = [t[3] for t in ds]
    v4 = [t[4] for t in ds]
    v7 = [t[7] for t in ds]

    v0max = max(v0)
    v0min = min(v0)

    v3max = max(v3)
    v3min = min(v3)

    v4max = max(v4)
    v4min = min(v4)

    v7max = max(v7)
    v7min = min(v7)

    def norm(t, min, max):
        return (t - min)/(max-min)

    for t in ds:
        t[0] = norm(t[0], v0min, v0max)
        t[3] = norm(t[3], v3min, v3max)
        t[4] = norm(t[4], v4min, v4max)
        t[7] = norm(t[7], v7min, v7max)

    return ds

def main():
    # Read in the dataset
    df = pd.read_csv('binByMean.csv', sep=',', header=None)

    avg_acc = []
    
    # Run 100 times to get average accuracy after many reshuffles
    for epoch in range(100):
        s = df.values # Values from the dataset (s is now a 2d list)
        s= shuffle(s) # Shuffle the values around as dataset is ordered by class column
        s = normalize(s)

        # Split 70%/30%
        train = s[:240]
        test = s[240:]

        k = 15

        predictions = []
        for x in range(len(test)):
            dstindex = buildDistanceMatrix(train, test[x])
            closest = getNearest(k, dstindex)
            result = getValue(closest, train)
            predictions.append(result)

        accuracy = acc(predictions, test) * 100.0
        print("Epoch: " + str(epoch))
        print("Accuracy: " + "{0:.5f}".format(accuracy) + "%")

        avg_acc.append(accuracy)

    avg = sum(avg_acc)/len(avg_acc)
    print(avg)

if __name__=="__main__":
    main()