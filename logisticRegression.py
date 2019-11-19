import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing

def normalize(ds):
    """ 
    Normalize features 0, 3, 4, and 7
    """

    # Feature 0
    v0 = ds[[0]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    v0_scaled = min_max_scaler.fit_transform(v0)
    df = pd.DataFrame(v0_scaled)
    ds[[0]] = df

    # Feature 3
    v3 = ds[[3]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    v3_scaled = min_max_scaler.fit_transform(v3)
    df = pd.DataFrame(v3_scaled)
    ds[[3]] = df

    # Feature 4
    v4 = ds[[4]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    v4_scaled = min_max_scaler.fit_transform(v4)
    df = pd.DataFrame(v4_scaled)
    ds[[4]] = df

    # Feature 7
    v7 = ds[[7]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    v7_scaled = min_max_scaler.fit_transform(v7)
    df = pd.DataFrame(v7_scaled)
    ds[[7]] = df

    return ds

def main():
    
    # Read the dataset
    df = pd.read_csv('binByMean.csv', sep=',', header=None)
    randomized_df = normalize(df)
    avg_acc = []

    # Run for 1000 epochs to get an average
    for epoch in range(1000):
    
        # Dataset is sorted by predicted attribute
        # Shuffle so splitting into training and test sets contain a mix
        randomized_df = shuffle(df)

        # Feature columns are 0 to 12
        feature_cols = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        # X = features
        X = randomized_df[feature_cols]
    
        # y = predicted attribute
        y = randomized_df[15]

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Build the model using the liblinear solver
        logreg = LogisticRegression(solver='liblinear')

        # Fit the model according to given training data
        logreg.fit(X_train, y_train)

        # The prediction
        y_pred=logreg.predict(X_test)

        # Get the accuracy
        accuracy = metrics.accuracy_score(y_test, y_pred)

        avg_acc.append(accuracy)

    # Compute the average accuracy over 1000 epochs
    avg = sum(avg_acc)/len(avg_acc)

    print(avg)

if __name__=="__main__":
    main()
