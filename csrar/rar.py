# Created by Daniel Thevessen

from hics.result_storage import DefaultResultStorage
from hics.incremental_correlation import IncrementalCorrelation
from csrar.rar_search import RaRSearch
import pandas as pd
import numpy as np


class RaR:
  def __init__(self, data):
    self.data = data
    self.correlation = None
    self.feature_ranking = None

  def run(self, target, k=5, runs=None, cost_matrix=None):
    if cost_matrix:
        assert cost_matrix is pd.DataFrame and len(cost_matrix.index) == len(cost_matrix.columns), \
            'Cost matrix needs to be a square-form pandas.DataFrame!'
        uniques = set(np.unique(self.data[target]))
        assert uniques == set(cost_matrix.columns) and uniques == set(cost_matrix.index)

    if runs:
        runs = RaRSearch.monte_carlo_fixed(runs=runs)

    if self.correlation is None or target != self.correlation.target:
        input_features = [ft for ft in self.data.columns.values if ft != target]
        storage = DefaultResultStorage(input_features)
        self.correlation = IncrementalCorrelation(self.data, target, storage)

    rar_search = RaRSearch(self.correlation, k=k, monte_carlo=runs)
    self.feature_ranking = rar_search.select_features()

    for (index, rank) in enumerate(self.feature_ranking):
        print('{}. {}'.format(index + 1, rank[0]))
