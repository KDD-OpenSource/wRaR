# Created by Marcus Pappik

import numpy as np
import pandas as pd
from hics.scored_slices import ScoredSlices


class AbstractResultStorage:

    def update_relevancies(self, new_relevancies: pd.DataFrame):
        raise NotImplementedError()

    def update_redundancies(self, new_redundancies: pd.DataFrame):
        raise NotImplementedError()

    def update_bivariate_redundancies(self, new_redundancies: pd.DataFrame, new_weights: pd.DataFrame):
        raise NotImplementedError()

    def update_slices(self, new_slices: dict()):
        raise NotImplementedError()

    def get_bivariate_redundancies(self):
        raise NotImplementedError()

    def get_redundancies(self):
        raise NotImplementedError()

    def get_relevancies(self):
        raise NotImplementedError()

    def get_slices(self):
        raise NotImplementedError()


class DefaultResultStorage(AbstractResultStorage):

    def __init__(self, features: list()):
        self.relevancies = pd.DataFrame(columns=['relevancy', 'iteration'])
        self.redundancies = pd.DataFrame(columns=['redundancy', 'iteration'])

        redundancy_dict = {
            'redundancy': pd.DataFrame(data=0, columns=features, index=features),
            'weight': pd.DataFrame(data=0, columns=features, index=features)
        }
        self.bivariate_redundancies = pd.Panel(redundancy_dict)

        self.slices = {}

    def update_relevancies(self, new_relevancies: pd.DataFrame):
        self.relevancies = new_relevancies

    def update_redundancies(self, new_redundancies: pd.DataFrame):
        """Updates redundancies by averaging it with existing values
        """
        current_redundancies = self.redundancies

        new_index = [index for index in new_redundancies.index
                     if index not in current_redundancies.index]

        redundancy_apppend = pd.DataFrame(data=0, index=new_index,
                                          columns=current_redundancies.columns)
        current_redundancies = current_redundancies.append(redundancy_apppend)

        current_redundancies.loc[new_redundancies.index, 'redundancy'] = \
            (current_redundancies.iteration / (current_redundancies.iteration + new_redundancies.iteration)) \
            * current_redundancies.redundancy \
            + \
            (new_redundancies.iteration / (current_redundancies.iteration + new_redundancies.iteration)) \
            * new_redundancies.redundancy
        current_redundancies.loc[new_redundancies.index, 'iteration'] += new_redundancies.iteration

        self.redundancies = current_redundancies

    def update_bivariate_redundancies(self, new_redundancies: pd.DataFrame, new_weights: pd.DataFrame):
        current_weights = current_weights.loc[new_weights.index, new_weights.columns]
        current_redundancies = current_redundancies.loc[new_redundancies.index, new_redundancies.columns]

        current_redundancies[current_weights < 1] = np.inf

        current_redundancies = np.minimum(new_redundancies, current_redundancies)
        current_weights += new_weights

        current_redundancies[current_weights < 1] = 0

        self.bivariate_redundancies.redundancy = current_redundancies
        self.bivariate_redundancies.weight = current_weights

    def update_slices(self, new_slices):
        self.slices = new_slices

    def get_bivariate_redundancies(self):
        return self.bivariate_redundancies.redundancy, self.bivariate_redundancies.weight

    def get_redundancies(self):
        return self.redundancies

    def get_relevancies(self):
        return self.relevancies

    def get_slices(self):
        return self.slices
