# Created by Daniel Thevessen

from hics.result_storage import DefaultResultStorage
from hics.incremental_correlation import IncrementalCorrelation
from csrar.rar_search import RaRSearch
import numpy as np
import pandas as pd


class RaR:
  def __init__(self, data):
    self.data = data
    self.correlation = None
    self.feature_ranking = None

  def run(self, target, k=5, runs=None, split_iterations=3, cost_matrix=None, compensate_imbalance=False, weight_mod=1):
    # if cost_matrix:
    #     assert cost_matrix is pd.DataFrame and len(cost_matrix.index) == len(cost_matrix.columns), \
    #         'Cost matrix needs to be a square-form pandas.DataFrame!'
    #     uniques = set(np.unique(self.data[target]))
    #     assert uniques == set(cost_matrix.columns) and uniques == set(cost_matrix.index)

    # Generate cost matrix compensating class imbalance
    if compensate_imbalance:
        values, counts = np.unique(self.data[target], return_counts=True)
        ci_matrix = pd.DataFrame(columns=values)
        for value, count in zip(values, counts):
            weighting = (len(self.data) / count) ** weight_mod
            ci_matrix[value] = [weighting]

        if cost_matrix:
           assert cost_matrix.columns == values
        else:
           cost_matrix = pd.DataFrame(1, index=[0], columns=values)
        cost_matrix = ci_matrix * cost_matrix
        print('Generated cost matrix:\n{}\nOverall cost matrix:\n{}'.format(ci_matrix, cost_matrix))

    if runs:
        runs = RaRSearch.monte_carlo_fixed(runs=runs)

    if self.correlation is None or target != self.correlation.target:
        input_features = [ft for ft in self.data.columns.values if ft != target]
        storage = DefaultResultStorage(input_features)
        self.correlation = IncrementalCorrelation(self.data, target, storage,
                                                  cost_matrix=(cost_matrix if compensate_imbalance else None),
                                                  weight_mod=weight_mod)

    rar_search = RaRSearch(self.correlation, k=k, monte_carlo=runs,
                           split_iterations=split_iterations, cost_matrix=cost_matrix)
    self.feature_ranking = rar_search.select_features()

    for (index, rank) in enumerate(self.feature_ranking):
        print('{}. {} with a score of {}'.format(index + 1, rank[0], rank[1]))
