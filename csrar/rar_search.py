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
    self.correlation.update_redundancies(k=self.k, runs=self.monte_carlo(dim))
    return self._calculate_ranking()

  def _calculate_ranking(self):
    feature_relevances = self._calculate_single_feature_relevance(self.correlation.features,
                                                                  self.correlation.result_storage.relevancies.relevancy)

    available = set(self.correlation.features)
    selected = set()

    while available:  # not empty
      def score(feature):
        relevance = feature_relevances[feature]
        # Redundancy given already selected features
        redundancy = self._calculate_redundancy(feature, list(map(lambda f: f[0], selected)))
        # Rank features using f-score on (1-redundancy) and relevancy
        return (feature, 2 * (1 - redundancy) * relevance / ((1 - redundancy) + relevance))
      scores = map(score, available)

      nextBest = max(scores, key=lambda s: s[1])

      selected.add((nextBest[0], len(selected) + 1))
      available.remove(nextBest[0])

    sorted_ranking = sorted(selected, key=lambda f: f[1])
    return sorted_ranking

  def _calculate_redundancy(self, feature, subset):
    if not subset:
      return 0
    else:
      redundancies = self.correlation.result_storage.redundancies.redundancy

      subsets = [i for i in redundancies.index if i[1] == feature]
      admissible = [s for s in subsets if len(set(subset).intersection(set((s,)))) > 0]
      admissible.sort(lambda s: redundancies[s], reverse=True)

      def is_justified(head, tail):
        elementsInSet = set((head, )).intersection(set(subset))
        size = len(elementsInSet)
        return not any(len(set((s, )).intersect(elementsInSet)) == size for s in tail)

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
