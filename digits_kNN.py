from sklearn import datasets
from playML.kNN import KNNClassifier
from playML.model_selection import train_test_split
from playML.metrics import accuracy_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

knn_clf = KNNClassifier(k=3)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)

AS01 = accuracy_score(y_test, y_predict)
AS02 = knn_clf.score(X_test, y_test)

print(AS01)
print(AS02)

