from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

Vehicle = pd.read_csv('./Vehicle.csv')
X = Vehicle.iloc[:, 0:18]
y = Vehicle.iloc[:, 19:20]

k = 6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = y_train.values.ravel()
weights = np.full([X_train.shape[0], k], 1, dtype = 'float32')

def knn_weights(array):
    return weights

models = [KNeighborsClassifier(n_neighbors=k, weights = knn_weights), KNeighborsClassifier(n_neighbors=k, weights = knn_weights)]

def update_weighted(y_pred):
    diff_indices = np.where(y_pred != y_train)
    for index in np.nditer(diff_indices):
        weights[index] = weights[index]*2222

def main():
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        y_pred = models[i].predict(X_train)
        update_weighted(y_pred)
        accuracy = accuracy_score(y_pred, y_train)
        print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()