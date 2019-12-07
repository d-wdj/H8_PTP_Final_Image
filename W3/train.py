# -*- coding: utf-8 -*-

from time import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

model = RandomForestClassifier(
        criterion='entropy',
        n_estimators=10,
        random_state=1111
        )

vectorizer = CountVectorizer(stop_words='english')

def load_data(filename, sep='\t', columns=['labels', 'message']):
    """Load dataset"""
    data = pd.read_csv(filename, sep=sep, names=columns)
    X = data['message'].values
    y = data['labels'].values
    
    return X, y

if __name__ == "__main__":
    print ("Load data...")
    X, y = load_data("SMSSpamCollection.tsv")
    print ("Vectorising set...")
    bow = vectorizer.fit_transform(X)
    print ("Training classifier model...")
    kfold = KFold(n_splits=5, random_state=111)
    model_acc_score = []
    for z, (trainIdx, testIdx) in enumerate(kfold.split(bow)):        
        X_train, X_test = bow[trainIdx], bow[testIdx]
        y_train, y_test = y[trainIdx], y[testIdx]
        t0 = time()
        model.fit(X_train, y_train)
        print (f"Training fold{z+1} completed in {time()-t0:3f}s")
        model_acc_score.append(model.score(X_test, y_test))
    mean_score, std_score = np.mean(model_acc_score), np.std(model_acc_score)
    print (f"Average accuracy score: {mean_score:.5f} ± {std_score:.5f}")
    

    