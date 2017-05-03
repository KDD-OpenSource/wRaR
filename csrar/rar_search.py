# Created by Daniel Thevessen

from math import factorial, ceil, log


class RaRSearch:

  # TODO: RaR Params for non-fixed Monte-Carlo
  def __init__(hics, k=5, monte_carlo=mc_fixed_function(runs=50)):
    self.hics = hics
    self.k = k
    self.monte_carlo_count = monte_carlo_count

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

  def _calculate_ranking:
    pass

  def select_features(self):
    dim = len(hics.data.columns)

    hics.update_multivariate_relevancies(k=self.k, runs=self.monte_carlo(dim))
    hics.update_redundancies(k=self.k, runs=self.monte_carlo(dim))
