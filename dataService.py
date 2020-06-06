import pandas as pd
import numpy


class ImportData:
    def __init__(self,
                 dataPath='data/winequality-red.csv',
                 columnsPath='data/winequality.names'):
        self.dataset_path = dataPath
        self.columns_path = columnsPath

    def importNamesOfColumns(self)-> numpy.ndarray:
        columns = pd.read_csv(self.columns_path, sep=';', comment='#', header=None).to_numpy()

        return numpy.concatenate(columns, axis=0)

    def importColumns(self, selected_columns_names: numpy.ndarray)-> numpy.ndarray:

        columns_names = self.importNamesOfColumns()

        data = pd.read_csv(self.dataset_path, sep=';', names=columns_names, usecols=selected_columns_names)

        return data.values

    def importColumnsWithoutClass(self) -> numpy.ndarray:
        columns_names = self.importNamesOfColumns()
        result = numpy.take(columns_names, range(-1, 11))

        return result

    def importAllData(self) -> numpy.ndarray:

        columns_names = self.importNamesOfColumns()
        usecols = self.importColumnsWithoutClass()

        data = pd.read_csv(self.dataset_path, sep=';', index_col=-1, names=columns_names, usecols=usecols)
        return data.values

    def importTrainData(self) -> numpy.ndarray:

        columns_names = self.importNamesOfColumns()
        usecols = self.importColumnsWithoutClass()

        data = pd.read_csv(self.dataset_path, sep=';', index_col=-1, names=columns_names,
                           usecols=usecols, nrows=2000)

        return data.values

    def importColumnsTrain(self, selected_columns_names: numpy.ndarray)-> numpy.ndarray:

        columns_names = self.importNamesOfColumns()

        data = pd.read_csv(self.dataset_path, sep=';', names=columns_names,
                           usecols=selected_columns_names, nrows=2000)

        return data.values