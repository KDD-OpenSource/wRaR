import numpy as np
import pandas as pd
from hics.slice_similarity import continuous_similarity_matrix, categorical_similarity_matrix
from hics.slice_selection import select_by_similarity


class ScoredSlices:
	def __init__(self, categorical, continuous, to_keep = 5, threshold = None):
		self.continuous = pd.Panel({feature : pd.DataFrame(columns = ['end', 'start'])
			for feature in continuous})

		self.categorical = pd.Panel({feature['name'] : pd.DataFrame(columns = feature['values'])
			for feature in categorical})

		self.scores = pd.Series()
		self.to_keep = to_keep
		
		if threshold == None:
			self.threshold = pow(0.6, len(categorical) + len(continuous))
		else:
			self.threshold = threshold

	def add_slices(self, slices):
		if isinstance(slices, dict):
			self.add_from_dict(slices)

		else:
			self.add_from_object(slices)

	def add_from_object(self, slices):
		temp_continuous = {}
		temp_categorical = {}

		self.scores = self.scores.append(pd.Series(slices.scores)).sort_values(ascending = False, inplace = False)

		for feature, df in slices.continuous.iteritems():
			temp_continuous[feature] = pd.concat([self.continuous[feature], df], ignore_index = True)
			temp_continuous[feature] = temp_continuous[feature].loc[self.scores.index, :].reset_index(drop = True)

		for feature, df in slices.categorical.iteritems():
			temp_categorical[feature] = pd.concat([self.categorical[feature], df], ignore_index = True)
			temp_categorical[feature] = temp_categorical[feature].loc[self.scores.index, :].reset_index(drop = True)

		self.scores.reset_index(drop = True, inplace = True)
		self.continuous = pd.Panel(temp_continuous)
		self.categorical = pd.Panel(temp_categorical)

	def add_from_dict(self, slices):
		temp_continuous = {}
		temp_categorical = {}

		new_scores = pd.Series(slices['scores'])
		self.scores = self.scores.append(new_scores, ignore_index = True).sort_values(ascending = False, inplace = False)

		for feature in self.continuous.items.values:
			content = pd.DataFrame(slices['features'][feature])
			temp_continuous[feature] = pd.concat([self.continuous[feature], content], ignore_index = True)
			temp_continuous[feature] = temp_continuous[feature].loc[self.scores.index, :].reset_index(drop = True)

		for feature in self.categorical.items.values:
			content = pd.DataFrame(slices['features'][feature], columns = self.categorical[feature].columns.values)
			temp_categorical[feature] = pd.concat([self.categorical[feature], content], ignore_index = True)
			temp_categorical[feature] = temp_categorical[feature].loc[self.scores.index, :].reset_index(drop = True)

		self.scores.reset_index(drop = True, inplace = True)
		self.continuous = pd.Panel(temp_continuous)
		self.categorical = pd.Panel(temp_categorical)

	def select_slices(self, similarity):
		indices = list(range(len(similarity)))
		selected = []

		for i in range(self.to_keep):
			if not indices:
				break

			selected.append(indices[0])
			selection = indices[0]

			indices = [index for index in indices if similarity[selection, index] < self.threshold]

		return selected

	def reduce_slices(self):
		if not self.continuous.empty:
			continuous_similarity = continuous_similarity_matrix(self.continuous)
		else:
			continuous_similarity = np.ones((len(self.scores), len(self.scores)))

		if not self.categorical.empty:
			categorical_similarity = categorical_similarity_matrix(self.categorical)
		else:
			categorical_similarity = np.ones((len(self.scores), len(self.scores)))

		similarity = continuous_similarity * categorical_similarity

		selected = self.select_slices(similarity)

		if not self.categorical.empty:
			self.categorical = self.categorical[:, selected, :]
			self.categorical = pd.Panel({name : content.reset_index(drop = True) 
				for name, content in self.categorical.iteritems()})
		
		if not self.continuous.empty:
			self.continuous = self.continuous[:, selected, :]
			self.continuous = pd.Panel({name : content.reset_index(drop = True) 
				for name, content in self.continuous.iteritems()})

		self.scores = self.scores.loc[selected].reset_index(drop = True)