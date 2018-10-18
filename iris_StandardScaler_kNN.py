import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from playML.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
scale_X_test = standardScaler.transform(X_test)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
KN_iris_score = knn_clf.score(scale_X_test, y_test)

print(KN_iris_score)
print(standardScaler.mean_)
print(standardScaler.scale_)




