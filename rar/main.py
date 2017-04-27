# Created by Daniel Thevessen


class RaR:
  def __init__(self, data, alpha, iterations, continuous_divergence=KS, categorical_divergence=KLD):
    self.iterations = iterations
    self.alpha = alpha
    self.data = data
    self.categorical_divergence = categorical_divergence
    self.continuous_divergence = continuous_divergence
    self.sorted_indices = pd.DataFrame()
    self.distributions = {}

    self.types = {}
    self.values = {}
    for column in self.data.columns.values:
      unique_values = np.unique(self.data[column])

      if self.data[column].dtype == 'object':
        self.types[column] = 'categorical'
        self.values[column] = unique_values

      elif len(unique_values) < 15:
        self.types[column] = 'categorical'
        self.values[column] = unique_values

      else:
        self.types[column] = 'continuous'
