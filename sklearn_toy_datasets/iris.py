import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target = pd.DataFrame(data=iris.target, columns=['class'])

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, stratify=target)

    # for c in df.columns:
    #     print(df[c].describe())

    # data = pd.concat((X_train, y_train), axis=1)
    # data.plot.scatter(x='sepal length (cm)', y='class').figure.show()

    lr = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=100000, l1_ratio=0.5)
    lr.fit(X_train, y_train.values.ravel())
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
