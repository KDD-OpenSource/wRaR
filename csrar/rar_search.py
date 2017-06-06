# Created by Daniel Thevessen

from math import factorial, ceil, log
from csrar.relevance_optimizer import RelevanceOptimizer
import collections


class RaRSearch(RelevanceOptimizer):

  # TODO: RaR Params for non-fixed Monte-Carlo
  def __init__(self, correlation, k=5, monte_carlo=None):
    self.correlation = correlation
    self.k = k
    self.monte_carlo = monte_carlo
    if monte_carlo is None:
      # Estimate for how many runs are necessary for good relevance "coverage"
      n = len(correlation.features)
      self.monte_carlo = RaRSearch.monte_carlo_fixed(runs=n)

  def monte_carlo_fixed(runs):
    def _mc_fixed(dim):
      return runs
    return _mc_fixed

  def _nCr(n, k):
    factorial(n) // factorial(k) // factorial(n - k)

  def monte_carlo_adaptive(runs, m, beta, min):
    def _mc_adaptive(dim):
      n = ceil(log(beta) / log(1 - _nCr(dim - m, k - m) / _nCr(dim, k)))
      return max(n, min)
    return _mc_adaptive

  def select_features(self):
    dim = len(self.correlation.features)

    self.correlation.update_multivariate_relevancies(k=self.k, runs=self.monte_carlo(dim))
    return self._calculate_ranking()

  def _calculate_ranking(self):
    feature_relevances = self._calculate_single_feature_relevance(self.correlation.features,
                                                                  self.correlation.result_storage.relevancies.relevancy)
    feature_redundancies = self._calculate_redundancies(self.correlation.features, feature_relevances)

    def score(feature):
      relevance = feature_relevances[feature]
      redundancy = feature_redundancies[feature]
      # Rank features using f-score on (1-redundancy) and relevancy
      return (feature, 2 * (1 - redundancy) * relevance / ((1 - redundancy) + relevance))
    scores = map(score, self.correlation.features)

    sorted_ranking = sorted(selected, key=lambda f: f[1], reverse=True)
    return sorted_ranking

  def _calculate_redundancies(self, features, relevances):
    sorted_features = sorted(features, key=lambda f: relevances[f])

    redundancies = {sorted_features[0]: 0}
    for i in range(1, len(sorted_features)):
      redundancy = self.correlation.subspace_contrast.calculate_contrast(sorted_features[:i], sorted_features[i])
      redundancies[sorted_features[i]] = redundancy

  def _calculate_redundancy(self, feature, subset):
    if not subset:
      return 0
    else:
      redundancies = self.correlation.result_storage.redundancies.redundancy

      subsets = [i for i in redundancies.index if i[1] == feature]
      admissible = [s for s in subsets if len(set(subset).intersection(set((s,)))) > 0]
      admissible.sort(key=lambda s: redundancies[s], reverse=True)

      def is_justified(head, tail):
        elementsInSet = set((head, )).intersection(set(subset))
        size = len(elementsInSet)
        return not any(elementsInSet.issubset(set((s, ))) for s in tail)

      def select_justified(subsets):
        if not subsets:
          return 0
        head, *tail = subsets
        if is_justified(head, tail):
          return redundancies[s]
        else:
          select_justified(tail)

      max_justified = select_justified(admissible)
      return max_justified
