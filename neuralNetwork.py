from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from dataService import ImportData

import numpy as np

if __name__ == "__main__":
    dataSet = ImportData()

    x_train, x_test, y_train, y_test = \
        train_test_split(dataSet.importAllData(),
                         dataSet.importColumns
                         (np.array(['quality'])),
                         test_size=0.2, random_state=13)

    NN = MLPClassifier(solver='adam', alpha=0.0001,
                       hidden_layer_sizes=(1, 50),
                       random_state=1, max_iter=2000, verbose=1).fit(x_train, y_train.ravel())
    predictions = NN.predict(x_train)
    print(predictions)
    print(round(NN.score(x_test, y_test.ravel()), 4))