# Created by Daniel Thevessen

from hics.result_storage import DefaultResultStorage
from hics.incremental_correlation import IncrementalCorrelation
from csrar.rar_search import RaRSearch


class RaR:
  def __init__(self, data):
    self.data = data

  def run(self, target, k=5, runs=None):
    if runs:
        runs = RaRSearch.monte_carlo_fixed(runs=runs)

    input_features = [ft for ft in self.data.columns.values if ft != target]
    storage = DefaultResultStorage(input_features)
    correlation = IncrementalCorrelation(self.data, target, storage)

    rar_search = RaRSearch(correlation, k=k, monte_carlo=runs)
    feature_ranking = rar_search.select_features()

    # TODO
    for (index, rank) in enumerate(feature_ranking):
        print('{}. {} with a score of {}'.format(index + 1, rank[0], rank[1]))
