from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=666)


param_grid = [
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1, 11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1, 11)],
        'p':[i for i in range(1,6)],
        'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }
]

grid_knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(grid_knn_clf, param_grid, n_jobs=1, verbose=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_index_)