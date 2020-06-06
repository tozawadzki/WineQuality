from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from dataService import ImportData
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataSet = ImportData()
x: np.ndarray = dataSet.importTrainData()
y: np.ndarray = dataSet.importColumnsTrain(
        np.array(['quality']))
svc = SVC(kernel="linear")
rfe = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfe.fit(x, y.ravel())

print("Optymalna liczba cech : %d" % rfe.n_features_)
plt.figure()
plt.xlabel("Liczba cech")
plt.ylabel("Wynik")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()

