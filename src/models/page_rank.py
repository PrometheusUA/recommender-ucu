import os
import pandas as pd

import numpy as np

import sys
sys.path.append('../../')
from src.models._base import BaseModel
from src.utils import read_json_df
import random
from typing import Iterable, Union
import networkx as nx
import sklearn.metrics.pairwise
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

class PageRankModel(BaseModel):
    def __init__(self, target_col: str = 'stars', unique_stars: list = [1., 2., 3., 4., 5.,]) -> None:
        super().__init__(target_col, unique_stars)
        self.fitted = False
        self.review_train_df = None

    def create_matrix(self,review_df):
        '''
        This method of creating a graph is quite effective,
        in it we get the distance by taking the Euclidean difference.
        If one of the guests visited the restaurant and the other did not,
        then we take the maximum possible difference,
        if both guests visited the same restaurants,
        we compare their ratings. 
        If they both did not visit the restaurant, 
        then the difference is zero.
        Therefore, I believe that this method is effective and without changes.'''
        matrix = sparse.csr_matrix((review_df.stars.astype(float),
                               (review_df.user_id.cat.codes,
                                review_df.business_id.cat.codes))).toarray()
        
        dist = sklearn.metrics.pairwise.pairwise_distances(matrix,matrix)
        self.G = nx.from_numpy_array(dist)
    def fit(self, 
            review_train_df: pd.DataFrame, 
            user_df: pd.DataFrame, 
            business_df: pd.DataFrame) -> None:  # `user_df` and `business_df` can be different between fit and predict

        self.train_df = review_train_df.copy()
        self.train_df['user_id'] = self.train_df['user_id'].astype("category")
        self.train_df['business_id'] = self.train_df['business_id'].astype("category")
        self.train_df['user_codes'] = self.train_df['user_id'].cat.codes
        self.train_df['business_codes'] = self.train_df['business_id'].cat.codes
        self.create_matrix(self.train_df)

        # If we don't have user_id in train return best
        self.sorted = self.train_df.groupby('business_id').stars.mean().sort_values().reset_index()['business_id'].to_numpy()[::-1] 

    def predict(self, 
                review_val_df: pd.DataFrame, 
                user_df: pd.DataFrame, 
                business_df: pd.DataFrame, 
                predict_per_user: int = 10) -> Union[np.ndarray, pd.Series]:
        # Page rank could not predict stars.
        stars = np.ones((len(review_val_df)))
        total_df = pd.Series()
        for i in user_df.user_id:
            idx = self.train_df[self.train_df.user_id == i].user_id.cat.codes.max()
            if pd.isnull(idx):
                df = pd.Series(self.sorted[:predict_per_user],index=[i]*predict_per_user)
                total_df = pd.concat([total_df,df])
            else:
                best = nx.pagerank(self.G,personalization={idx:100000})
                # Find most similar users
                ratings = self.train_df[['user_id','business_id','stars']].copy()

                # Add weights to calculate best business
                ratings['weight'] = self.train_df['user_codes'].apply(lambda x:best[x])
                ratings['stars']*=ratings['weight']

                # Sort by weighted sum
                ratings = ratings.groupby('business_id').stars.sum().sort_values()[::-1].reset_index()['business_id']
                df = pd.Series(ratings[:predict_per_user].to_list(),index=[i]*predict_per_user)
                
                total_df = pd.concat([total_df,df])
        return pd.Series(stars),total_df