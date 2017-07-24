# Created by Daniel Thevessen

import gurobipy as gb
import sys
import pandas as pd
from collections import defaultdict


class RelevanceOptimizer:

  gb.setParam('OutputFlag', 0)

  def _calculate_single_feature_relevance(self, columns, relevances, cost_matrix=None):
    # New approach, keep class relevances separate
    if cost_matrix is not None:
      for index, df in relevances.items():
        df.index = [index]
      dataframe = pd.concat(relevances.values)
      classes = {col: dataframe[col] for col in dataframe.columns}
    else:
      classes = {'-1': relevances}

    single_relevances = defaultdict(int)
    for class_col, class_scores in classes.items():
      # print(class_col)
      m = gb.Model('rar')
      n = len(columns)
      max_score = max(class_scores)

      solver_variables = {}
      for col in columns:
        solver_variables[col] = m.addVar(name='x_' + col, vtype=gb.GRB.CONTINUOUS, lb=0, ub=max_score)

      vars_average = m.addVar(name='s', vtype=gb.GRB.CONTINUOUS)
      vars_sum = sum(solver_variables.values())

      m.addConstr(vars_average == (vars_sum / n))
      m.setObjective(vars_sum + self.__squared_dist(solver_variables.values(), vars_average), gb.GRB.MINIMIZE)
      m.update()

      for (subset, score) in class_scores.iteritems():
        objective_vars = map(lambda col: solver_variables[col], subset)
        objective_sum = sum(objective_vars)
        m.addConstr(objective_sum >= score)

      m.optimize()

      var_max = max(map(lambda v: v.x, solver_variables.values()))
      class_result = {k: (v.x / var_max) for k, v in solver_variables.items()}
      for k, v in solver_variables.items():
        single_relevances[k] += (v.x / var_max) * (1 if cost_matrix is None else cost_matrix[class_col][0])
        # print(str(v.x / var_max) + ' weighted to ' +
        # str((v.x / var_max) * (1 if cost_matrix is None else cost_matrix[class_col][0])))

    for k in single_relevances.keys():
      single_relevances[k] /= (1 if cost_matrix is None else cost_matrix.iloc[0].sum())

    return single_relevances

  def __squared_dist(self, variables, mean):
    return sum(map(lambda v: (v - mean) * (v - mean), variables))
