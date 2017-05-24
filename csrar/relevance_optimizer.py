# Created by Daniel Thevessen

import gurobipy as gb


class RelevanceOptimizer:

  gb.setParam('OutputFlag', 0)

  def _calculate_single_feature_relevance(self, columns, relevances):
    m = gb.Model('rar')
    n = len(columns)
    max_score = max(relevances)

    solver_variables = {}
    for col in columns:
      solver_variables[col] = m.addVar(name='x_' + col, vtype=gb.GRB.CONTINUOUS, lb=0, ub=max_score)

    vars_average = m.addVar(name='s', vtype=gb.GRB.CONTINUOUS)
    vars_sum = sum(solver_variables.values())

    m.addConstr(vars_average == (vars_sum / n))
    m.setObjective(vars_sum + self.__squared_dist(solver_variables.values(), vars_average), gb.GRB.MINIMIZE)
    m.update()

    for (subset, score) in relevances.iteritems():
      objective_vars = map(lambda col: solver_variables[col], subset)
      objective_sum = sum(objective_vars)
      m.addConstr(objective_sum >= score)

    m.optimize()

    var_max = max(map(lambda v: v.x, solver_variables.values()))
    return {k: (v.x / var_max) for k, v in solver_variables.items()}

  def __squared_dist(self, variables, mean):
    return sum(map(lambda v: (v - mean) * (v - mean), variables))
