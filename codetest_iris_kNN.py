import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from .model_selection import train_test_split
from .kNN import KNNClassifier
from .metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

my_knn_clf = KNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)

y_predict = my_knn_clf.predict(X_test)

accuracy_metrics = accuracy_score(y_test, y_predict)
accuracy_knn_score = my_knn_clf.score(X_test, y_test)

print('metrics：' , float(accuracy_metrics))
print('knn_score：' , float(accuracy_knn_score))
