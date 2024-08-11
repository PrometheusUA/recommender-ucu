import numpy as np
import math
import random
import copy
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
import pandas as pd
import pickle
from collections import defaultdict
from tqdm.autonotebook import tqdm


def dcg(scores):
	"""
		Returns the DCG value of the list of scores.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		
		Returns
		-------
		DCG_val: int
			This is the value of the DCG on the given scores
	"""
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in range(len(scores))
					])

def dcg_k(scores, k):
	"""
		Returns the DCG value of the list of scores and truncates to k values.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		k : int
			In the amount of values you want to only look at for computing DCG
		
		Returns
		-------
		DCG_val: int
			This is the value of the DCG on the given scores
	"""
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in range(len(scores[:k]))
					])

def ideal_dcg(scores):
	"""
		Returns the Ideal DCG value of the list of scores.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		
		Returns
		-------
		Ideal_DCG_val: int
			This is the value of the Ideal DCG on the given scores
	"""
	scores = [score for score in sorted(scores)[::-1]]
	return dcg(scores)

def ideal_dcg_k(scores, k):
	"""
		Returns the Ideal DCG value of the list of scores and truncates to k values.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		k : int
			In the amount of values you want to only look at for computing DCG
		
		Returns
		-------
		Ideal_DCG_val: int
			This is the value of the Ideal DCG on the given scores
	"""
	scores = [score for score in sorted(scores)[::-1]]
	return dcg_k(scores, k)

def single_dcg(scores, i, j):
	"""
		Returns the DCG value at a single point.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		i : int
			This points to the ith value in scores
		j : int
			This sets the ith value in scores to be the jth rank
		
		Returns
		-------
		Single_DCG: int
			This is the value of the DCG at a single point
	"""
	return (np.power(2, scores[i]) - 1) / np.log2(j + 2)

def compute_lambda(args):
	"""
		Returns the lambda and w values for a given query.
		Parameters
		----------
		args : zipped value of true_scores, predicted_scores, good_ij_pairs, idcg, query_key
			Contains a list of the true labels of documents, list of the predicted labels of documents,
			i and j pairs where true_score[i] > true_score[j], idcg values, and query keys.
		
		Returns
		-------
		lambdas : numpy array
			This contains the calculated lambda values
		w : numpy array
			This contains the computed w values
		query_key : int
			This is the query id these values refer to
	"""

	true_scores, predicted_scores, good_ij_pairs, idcg, query_key = args
	num_docs = len(true_scores)
	sorted_indexes = np.argsort(predicted_scores)[::-1]
	rev_indexes = np.argsort(sorted_indexes)
	true_scores = true_scores[sorted_indexes]
	predicted_scores = predicted_scores[sorted_indexes]

	lambdas = np.zeros(num_docs)
	w = np.zeros(num_docs)

	single_dcgs = {}
	for i,j in good_ij_pairs:
		if (i,i) not in single_dcgs:
			single_dcgs[(i,i)] = single_dcg(true_scores, i, i)
		single_dcgs[(i,j)] = single_dcg(true_scores, i, j)
		if (j,j) not in single_dcgs:
			single_dcgs[(j,j)] = single_dcg(true_scores, j, j)
		single_dcgs[(j,i)] = single_dcg(true_scores, j, i)


	for i,j in good_ij_pairs:
		z_ndcg = abs(single_dcgs[(i,j)] - single_dcgs[(i,i)] + single_dcgs[(j,i)] - single_dcgs[(j,j)]) / idcg
		rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
		rho_complement = 1.0 - rho
		lambda_val = z_ndcg * rho
		lambdas[i] += lambda_val
		lambdas[j] -= lambda_val

		w_val = rho * rho_complement * z_ndcg
		w[i] += w_val
		w[j] += w_val

	return lambdas[rev_indexes], w[rev_indexes], query_key

def group_queries(query_ids):
	"""
		Returns a dictionary that groups the documents by their query ids.
		Parameters
		----------
		training_data : Numpy array of lists
			Contains a list of document information. Each document's format is [relevance score, query index, feature vector]
		qid_index : int
			This is the index where the qid is located in the training data
		
		Returns
		-------
		query_indexes : dictionary
			The keys were the different query ids and teh values were the indexes in the training data that are associated of those keys.
	"""
	query_indexes = defaultdict(list)
	for index, qid in enumerate(query_ids):
		query_indexes[qid].append(index)

	return query_indexes

def get_pairs(scores):
	"""
		Returns pairs of indexes where the first value in the pair has a higher score than the second value in the pair.
		Parameters
		----------
		scores : list of int
			Contain a list of numbers
		
		Returns
		-------
		query_pair : list of pairs
			This contains a list of pairs of indexes in scores.
	"""

	query_pair = []
	for query_scores in scores:
		temp = sorted(query_scores, reverse=True)
		pairs = []
		for i in range(len(temp)):
			for j in range(len(temp)):
				if temp[i] > temp[j]:
					pairs.append((i,j))
		query_pair.append(pairs)
	return query_pair

class LambdaMART:

	def __init__(self, training_features=None, training_queries=None, training_scores=None, number_of_trees=5, learning_rate=0.1, tree_max_depth=5):
		"""
		This is the constructor for the LambdaMART object.
		Parameters
		----------
		training_data : list of int
			Contain a list of numbers
		number_of_trees : int (default: 5)
			Number of trees LambdaMART goes through
		learning_rate : float (default: 0.1)
			Rate at which we update our prediction with each tree
		tree_type : string (default: "sklearn")
			Either "sklearn" for using Sklearn implementation of the tree of "original" 
			for using our implementation
		"""

		self.training_features = training_features
		self.training_queries = training_queries
		self.training_scores = training_scores

		self.number_of_trees = number_of_trees
		self.learning_rate = learning_rate
		self.trees = []
		self.tree_max_depth = tree_max_depth

	def fit(self):
		"""
		Fits the model on the training data.
		"""

		predicted_scores = np.zeros(len(self.training_scores))
		query_indexes = group_queries(self.training_queries)
		query_keys = query_indexes.keys()
		true_scores = [self.training_scores[query_indexes[query]] for query in query_keys]
		good_ij_pairs = get_pairs(true_scores)

		# ideal dcg calculations
		idcg = [ideal_dcg(scores) for scores in true_scores]

		for k in tqdm(range(self.number_of_trees), desc="Tree"):
			lambdas = np.zeros(len(predicted_scores))
			w = np.zeros(len(predicted_scores))
			pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]
			
			pool = Pool()
			for lambda_val, w_val, query_key in pool.map(compute_lambda, zip(true_scores, pred_scores, good_ij_pairs, idcg, query_keys), chunksize=1):
				indexes = query_indexes[query_key]
				lambdas[indexes] = lambda_val
				w[indexes] = w_val
			pool.close()

			tree = DecisionTreeRegressor(max_depth=self.tree_max_depth)
			tree.fit(self.training_features, lambdas)
			self.trees.append(tree)
			prediction = tree.predict(self.training_features)
			predicted_scores += prediction * self.learning_rate


	def predict(self, queries, features, verbose: bool = False):
		"""
		Predicts the scores for the test dataset.
		Parameters
		----------
		data : Numpy array of documents
			Numpy array of documents with each document's format is [query index, feature vector]
		
		Returns
		-------
		predicted_scores : Numpy array of scores
			This contains an array or the predicted scores for the documents.
		"""
		query_indexes = group_queries(queries)
		predicted_scores = np.zeros(len(queries))
		for query in query_indexes if not verbose else tqdm(query_indexes, desc="Prediction user"):
			results = np.zeros(len(query_indexes[query]))
			for tree in self.trees:
				results += self.learning_rate * tree.predict(features[query_indexes[query]])
			predicted_scores[query_indexes[query]] = results
		return predicted_scores

	def validate(self, data, k):
		"""
		Predicts the scores for the test dataset and calculates the NDCG value.
		Parameters
		----------
		data : Numpy array of documents
			Numpy array of documents with each document's format is [relevance score, query index, feature vector]
		k : int
			this is used to compute the NDCG@k
		
		Returns
		-------
		average_ndcg : float
			This is the average NDCG value of all the queries
		predicted_scores : Numpy array of scores
			This contains an array or the predicted scores for the documents.
		"""
		data = np.array(data)
		query_indexes = group_queries(data, 1)
		average_ndcg = []
		predicted_scores = np.zeros(len(data))
		for query in query_indexes:
			results = np.zeros(len(query_indexes[query]))
			for tree in self.trees:
				results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
			predicted_sorted_indexes = np.argsort(results)[::-1]
			t_results = data[query_indexes[query], 0]
			t_results = t_results[predicted_sorted_indexes]
			predicted_scores[query_indexes[query]] = results
			dcg_val = dcg_k(t_results, k)
			idcg_val = ideal_dcg_k(t_results, k)
			ndcg_val = (dcg_val / idcg_val)
			average_ndcg.append(ndcg_val)
		average_ndcg = np.nanmean(average_ndcg)
		return average_ndcg, predicted_scores

	def save(self, fname):
		"""
		Saves the model into a ".lmart" file with the name given as a parameter.
		Parameters
		----------
		fname : string
			Filename of the file you want to save
		
		"""
		pickle.dump(self, open('%s.lmart' % (fname), "wb"), protocol=2)

	def load(self, fname):
		"""
		Loads the model from the ".lmart" file given as a parameter.
		Parameters
		----------
		fname : string
			Filename of the file you want to load
		
		"""
		model = pickle.load(open(fname , "rb"))
		self.training_data = model.training_data
		self.number_of_trees = model.number_of_trees
		self.tree_type = model.tree_type
		self.learning_rate = model.learning_rate
		self.trees = model.trees
