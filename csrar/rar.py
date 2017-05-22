# Created by Daniel Thevessen

from hics.result_storage import DefaultResultStorage
from hics.incremental_correlation import IncrementalCorrelation
from csrar.rar_search import RaRSearch


class RaR:
  def __init__(self, data):
    self.data = data

  def run(self, target):
    input_features = [ft for ft in self.data.columns.values if ft != target]
    storage = DefaultResultStorage(input_features)
    correlation = IncrementalCorrelation(self.data, target, storage)

    rar_search = RaRSearch(correlation)
    feature_ranking = rar_search.select_features()

    # TODO
    print(feature_ranking)
