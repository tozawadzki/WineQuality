from sklearn.ensemble import ExtraTreesClassifier
from dataService import ImportData

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



if __name__ == "__main__":

    dataService = ImportData()
x: np.ndarray = dataService.importTrainData()
y: np.ndarray = dataService.importColumnsTrain(
        np.array(['quality']))
name_of_columns: np.ndarray = dataService.importNamesOfColumns()
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=['fixed acidity', 'volatile acidity', 'citric acid',
                                                                'residual sugar', 'chlorides', 'free sulfur dioxide',
                                                                'total sulfur dioxide', 'density', 'ph', 'sulphates',
                                                                'alcohol'])
feat_importances.nlargest(11).plot(kind='barh')
plt.xlabel("Wpływ cech na ocenę")
plt.show()