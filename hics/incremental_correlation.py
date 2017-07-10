# Created by Marcus Pappik

import numpy as np
import pandas as pd
import sys
from random import randint
from hics.contrast_measure import HiCS
from hics.scored_slices import ScoredSlices
from hics.result_storage import DefaultResultStorage


class IncrementalCorrelation:
    def __init__(self, data, target, result_storage, iterations=10,
                 alpha=0.1, drop_discrete=False, cost_matrix=None, weight_mod=1):
        self.subspace_contrast = HiCS(data, alpha, iterations)

        self.target = target
        self.features = [str(ft) for ft in data.columns.values
                         if str(ft) != target]
        self.cost_matrix = cost_matrix
        self.weight_mod = weight_mod

        if drop_discrete:
            self.features = [ft for ft in self.features
                             if self.subspace_contrast.get_type(ft) != 'discrete']

        self.result_storage = result_storage

    def _update_relevancy_table(self, new_relevancies):
        """Updates relevancies by averaging it with existing values
        """
        current_relevancies = self.result_storage.get_relevancies()

        new_index = [index for index in new_relevancies.index
                     if index not in current_relevancies.index]

        relevancy_apppend = pd.DataFrame(data=0, index=new_index,
                                         columns=current_relevancies.columns)
        current_relevancies = current_relevancies.append(relevancy_apppend)

        current_relevancies.loc[new_relevancies.index, 'relevancy'] = \
            (current_relevancies.iteration / (current_relevancies.iteration + new_relevancies.iteration)) \
            * current_relevancies.relevancy \
            + \
            (new_relevancies.iteration / (current_relevancies.iteration + new_relevancies.iteration)) \
            * new_relevancies.relevancy
        current_relevancies.loc[new_relevancies.index, 'iteration'] += new_relevancies.iteration

        self.result_storage.update_relevancies(current_relevancies)

    def _update_redundancy_table(self, new_redundancies):
        # Calculate bivariate redundancies for fast access
        bivariate_redundancies = pd.DataFrame(data=np.inf, columns=self.features, index=self.features)
        bivariate_weights = pd.DataFrame(data=0, columns=self.features, index=self.features)
        for row in new_redundancies.itertuples():
            subspace, target = row.Index
            for ft in subspace:
                redundancy = min(bivariate_redundancies.loc[ft, target], row.redundancy)
                bivariate_redundancies.loc[ft, target] = redundancy
                bivariate_redundancies.loc[target, ft] = redundancy
                bivariate_weights.loc[ft, target] = bivariate_weights.loc[ft, target] + row.iteration
                bivariate_weights.loc[target, ft] = bivariate_weights.loc[target, ft] + row.iteration

        self.result_storage.update_bivariate_redundancies(bivariate_redundancies, bivariate_weights)
        # Save subset redundancies for further calculations
        self.result_storage.update_redundancies(new_redundancies)

    def _update_slices(self, new_slices):
        current_slices = self.result_storage.get_slices()

        for feature_set, slices_to_add in new_slices.items():
            if feature_set not in current_slices:
                current_slices[feature_set] = slices_to_add

            else:
                current_slices[feature_set].add_slices(slices_to_add)

            current_slices[feature_set].reduce_slices()

        self.result_storage.update_slices(current_slices)

    def _relevancy_dict_to_df(self, new_scores):
        indices = [tuple(index) for index in new_scores]
        scores = [score for index, score in new_scores.items()]
        new_relevancies = pd.DataFrame(data=scores, index=indices)
        return new_relevancies

    def _add_slices_to_dict(self, subspace, slices, slices_store):
        subspace_tuple = tuple(sorted(subspace))
        if subspace_tuple not in slices_store:
            categorical = [
                {
                    'name': ft,
                    'values': self.subspace_contrast.get_values(ft)
                }
                for ft in subspace if self.subspace_contrast.get_type(ft) == 'categorical']
            continuous = [ft for ft in subspace if self.subspace_contrast.get_type(ft) == 'continuous']
            slices_store[subspace_tuple] = ScoredSlices(categorical, continuous)

        slices_store[subspace_tuple].add_slices(slices)
        return slices_store

    def update_bivariate_relevancies(self, runs=5):
        """Reruns relevancy calculation of individual features toward the target. Result will be averaged with
        previous values to update the relevancy score
        """
        new_slices = {}
        new_scores = {(feature,): {'relevancy': 0, 'iteration': 0} for feature in self.features}

        for i in range(runs):
            for feature in self.features:
                subspace_tuple = (feature,)
                subspace_score, subspace_slices = self.subspace_contrast.calculate_contrast(
                    [feature], self.target, True, cost_matrix=self.cost_matrix, weight_mod=self.weight_mod)

                new_slices = self._add_slices_to_dict([feature], subspace_slices, new_slices)

                new_scores[subspace_tuple]['relevancy'] += subspace_score
                new_scores[subspace_tuple]['iteration'] += 1

        new_relevancies = self._relevancy_dict_to_df(new_scores)
        new_relevancies.relevancy /= new_relevancies.iteration

        self._update_relevancy_table(new_relevancies)
        self._update_slices(new_slices)

    def update_multivariate_relevancies(self, fixed_features=[], k=5, runs=5):
        """Reruns relevancy calculations for subsets (multivariate), updating existing values.
        Keyword arguments:
        fixed_features -- List of features to be included in every tested subset. Counts into k, leaving
                          k - len(fixed_features) variable components. If len(fixed_features) > k, only
                          fixed_features is used
        k -- the maximal subset size
        runs -- the number of iterations to run
        """
        new_slices = {}
        new_scores = {}

        feature_list = [feature for feature in self.features if feature not in fixed_features]
        max_k = k - len(fixed_features)
        max_k = min(max_k, len(feature_list))

        for i in range(runs):
            # Progress counter
            sys.stdout.write('\rRelevance: {:.2f}%     '.format(100 * i / runs))
            sys.stdout.flush()

            subspace = fixed_features[:]

            if 0 < max_k:
                end_index = randint(1, max_k)
                subspace += np.random.permutation(feature_list)[0:end_index].tolist()

            subspace_tuple = tuple(sorted(subspace))
            subspace_score, subspace_slices = self.subspace_contrast.calculate_contrast(
                subspace, self.target, True, cost_matrix=self.cost_matrix, weight_mod=1)

            if subspace_tuple not in new_scores:
                new_scores[subspace_tuple] = {'relevancy': 0, 'iteration': 0}

            new_scores[subspace_tuple]['relevancy'] += subspace_score
            new_scores[subspace_tuple]['iteration'] += 1

            new_slices = self._add_slices_to_dict(subspace, subspace_slices, new_slices)
        print('\rRelevance: 100.00%')

        new_relevancies = self._relevancy_dict_to_df(new_scores)
        new_relevancies.relevancy /= new_relevancies.iteration

        self._update_relevancy_table(new_relevancies)
        self._update_slices(new_slices)

    def update_redundancies(self, k=5, runs=10):
        """Reruns redundancy calculations, updating existing values.
        Keyword arguments:
        k -- the maximal subset size
        runs -- the number of iterations to run
        """
        # new_redundancies = pd.DataFrame(data=np.inf, columns=self.features, index=self.features)
        # new_weights = pd.DataFrame(data=0, columns=self.features, index=self.features)

        new_scores = {}

        k = min(k, len(self.features) - 1)
        for i in range(runs):
            number_features = randint(1, k)
            selected_features = np.random.permutation(self.features)[0:number_features + 1].tolist()
            target = selected_features[number_features]
            subspace = selected_features[0:number_features]

            subspace_feature_tuple = (tuple(sorted(subspace)), target)
            score = self.subspace_contrast.calculate_contrast(subspace, target, False)

            if subspace_feature_tuple not in new_scores:
                new_scores[subspace_feature_tuple] = {'redundancy': 0, 'iteration': 0}

            new_scores[subspace_feature_tuple]['redundancy'] += score
            new_scores[subspace_feature_tuple]['iteration'] += 1

        new_redundancies = self._relevancy_dict_to_df(new_scores)
        new_redundancies.redundancy /= new_redundancies.iteration

        self._update_redundancy_table(new_redundancies)
