# Created by Marcus Pappik

import numpy as np
import pandas as pd
from hics.divergences import KLD, KS
import math
from random import randint, shuffle
from collections import defaultdict


class HiCS:
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

            elif len(unique_values) < 30:
                self.types[column] = 'categorical'
                self.values[column] = unique_values

            else:
                self.types[column] = 'continuous'

        self.index_lookup = pd.DataFrame(data=np.arange(len(self.data)),
                                         index=self.data.index)  # Alternative to index.get_loc for custom indices

    def get_values(self, feature):
        if feature not in self.values:
            return False

        else:
            return self.values[feature]

    def get_type(self, feature):
        if feature not in self.types:
            return False

        else:
            return self.types[feature]

    def cached_marginal_distribution(self, feature):
        if feature not in self.distributions:
            values, counts = np.unique(self.data[feature], return_counts=True)
            self.distributions[feature] = pd.DataFrame({'value': values, 'count': counts,
                                                        'probability': counts / len(self.data)}).sort_values(by='value')
        return self.distributions[feature]

    def cached_sorted_indices(self, feature):
        if feature not in self.sorted_indices.columns:
            self.sorted_indices[feature] = self.data.sort_values(by=feature, kind='mergesort').index.values
        return self.sorted_indices[feature]

    def calculate_conditional_distribution(self, slice_conditions, target):
        filter_array = np.array([True] * len(self.data))

        for slice_condition in slice_conditions:
            temp_filter = np.array([False] * len(self.data))
            temp_filter[self.index_lookup.ix[slice_condition['indices']]] = True
            filter_array = np.logical_and(temp_filter, filter_array)

        values, counts = np.unique(self.data.loc[filter_array, target], return_counts=True)
        probabilities = counts / filter_array.sum()
        return pd.DataFrame({'value': values, 'count': counts, 'probability': probabilities}).sort_values(by='value')

    def create_categorical_condition(self, feature, instances_per_dimension):
        feature_distribution = self.cached_marginal_distribution(feature)
        shuffled_values = np.random.permutation(feature_distribution['value'])
        selected_values = []
        current_sum = 0

        # select random values of feature until there are >= instances_per_dimension samples with one of these values
        for value in shuffled_values:
            if current_sum < instances_per_dimension:
                selected_values.append(value)
                current_sum = current_sum + \
                    feature_distribution.loc[feature_distribution['value'] == value, 'count'].values
            else:
                break

        indices = self.data.loc[self.data[feature].isin(selected_values), ].index.tolist()
        return {'feature': feature, 'indices': indices, 'values': selected_values}

    def create_continuous_condition(self, feature, instances_per_dimension):
        sorted_feature = self.cached_sorted_indices(feature)
        max_start = len(sorted_feature) - instances_per_dimension
        start = randint(0, max_start)
        end = start + (instances_per_dimension - 1)

        start_value = self.data.loc[sorted_feature[start], feature]
        end_value = self.data.loc[sorted_feature[end], feature]
        indices = self.data.loc[np.logical_and(self.data[feature] >= start_value, self.data[feature] <= end_value), :] \
            .index.values.tolist()
        return {'feature': feature, 'indices': indices, 'from_value': start_value, 'to_value': end_value}

    def output_slices(self, score, conditions, slices):
        for condition in conditions:
            ft = condition['feature']

            if self.types[ft] == 'categorical':
                to_append = [1 * (value in condition['values']) for value in self.get_values(ft)]
                if ft in slices['features']:
                    slices['features'][ft].append(to_append)
                else:
                    slices['features'][ft] = [to_append]

            else:
                if ft in slices['features']:
                    slices['features'][ft]['from_value'].append(condition['from_value'])
                    slices['features'][ft]['to_value'].append(condition['to_value'])
                else:
                    slices['features'][ft] = {}
                    slices['features'][ft]['from_value'] = [condition['from_value']]
                    slices['features'][ft]['to_value'] = [condition['to_value']]

        slices['scores'].append(score)

        return slices

    def calculate_contrast(self, features, target, return_slices=False, cost_matrix=None, weight_mod=1):
        slices = {'features': {}, 'scores': []}

        instances_per_dimension = max(round(len(self.data) * math.pow(self.alpha, 1 / len(features))), 5)

        marginal_distribution = self.cached_marginal_distribution(target)

        sum_scores = 0
        iterations = self.iterations

        sum_binary_scores = defaultdict(lambda: {'sum': 0, 'count': 0})
        for iteration in range(iterations):
            slice_conditions = []

            for feature in features:
                if self.types[feature] == 'categorical':
                    slice_conditions.append(self.create_categorical_condition(feature, instances_per_dimension))
                else:
                    slice_conditions.append(self.create_continuous_condition(feature, instances_per_dimension))

            conditional_distribution = self.calculate_conditional_distribution(slice_conditions, target)

            if conditional_distribution.empty:
                iterations = iterations - 1
                continue

            if self.types[target] == 'categorical':
                class_scores, binary_scores = self.categorical_divergence(conditional_distribution,
                                                                          marginal_distribution)

                score = 0
                for value, d in class_scores.items():
                    score += d

                for value, d in binary_scores.items():
                    sum_binary_scores[value]['sum'] += d
                    sum_binary_scores[value]['count'] += 1
            else:
                score = self.continuous_divergence(marginal_distribution, conditional_distribution)

            sum_scores = sum_scores + score

            if return_slices:
                slices = self.output_slices(score, slice_conditions, slices)

        # Primary measure (correlation)
        avg_score = sum_scores / iterations

        # cost_matrix is not None if target is class, apply cost thend
        if cost_matrix is not None:
            binary_div_df = pd.DataFrame(columns=cost_matrix.columns, index=cost_matrix.index).fillna(0)
            for k, v in sum_binary_scores.items():
                binary_div_df.loc[0, k] = v['sum'] / v['count']
            div_score = (cost_matrix * binary_div_df).iloc[0].sum() / cost_matrix.iloc[0].sum()
            # avg_score = div_score
            if return_slices:
                return div_score, slices, binary_div_df
            else:
                return div_score, binary_div_df

        if return_slices:
            return avg_score, slices
        else:
            return avg_score
