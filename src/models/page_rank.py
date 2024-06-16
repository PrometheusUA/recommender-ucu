import warnings
from typing import Union

import pandas as pd
import numpy as np
import networkx as nx
import sklearn.metrics.pairwise
from scipy import sparse

from src.models._base import BaseModel


warnings.filterwarnings("ignore")

class PageRankModel(BaseModel):
    def __init__(self, target_col: str = 'stars', unique_stars: list = [1., 2., 3., 4., 5.,]) -> None:
        super().__init__(target_col, unique_stars)
        self.fitted = False
        self.review_train_df = None

    def create_matrix(self, review_df, stars_col_name = 'stars'):
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
        matrix = sparse.csr_matrix((review_df[stars_col_name].astype(float),
                               (review_df.user_id.cat.codes,
                                review_df.business_id.cat.codes)))
        
        cosine_sim = sklearn.metrics.pairwise.cosine_similarity(matrix, dense_output=False)
        cosine_sim.setdiag(0)
        self.G = nx.from_scipy_sparse_array(cosine_sim)

    def fit(self, 
            review_train_df: pd.DataFrame, 
            user_df: pd.DataFrame, 
            business_df: pd.DataFrame) -> None:  # `user_df` and `business_df` can be different between fit and predict

        self.train_df = review_train_df.copy()
        self.train_df['user_id'] = self.train_df['user_id'].astype("category")
        self.train_df['business_id'] = self.train_df['business_id'].astype("category")
        self.train_df['user_codes'] = self.train_df['user_id'].cat.codes
        self.train_df['business_codes'] = self.train_df['business_id'].cat.codes

        stars_users_min = self.train_df.groupby('user_id')['stars'].min()
        stars_users_max = self.train_df.groupby('user_id')['stars'].max()

        equal_mask = stars_users_min == stars_users_max

        stars_users_min.loc[equal_mask] -= 0.01
        stars_users_max.loc[equal_mask] += 0.01

        self.train_df['stars_user_min'] = self.train_df['user_id'].apply(stars_users_min.get)
        self.train_df['stars_user_max'] = self.train_df['user_id'].apply(stars_users_max.get)

        self.train_df['stars_normalized'] = (self.train_df['stars'] - self.train_df['stars_user_min']) / (self.train_df['stars_user_max'] - self.train_df['stars_user_min'])

        self.create_matrix(self.train_df, stars_col_name='stars_normalized')

        # If we don't have user_id in train return best
        self.sorted = self.train_df.groupby('business_id').stars_normalized.mean().sort_values().reset_index()['business_id'].to_numpy()[::-1]
        self.business_median_stars = self.train_df.groupby('business_id').stars.median()
        self.median_stars = self.train_df.stars.median()

    def predict(self, 
                review_val_df: pd.DataFrame, 
                user_df: pd.DataFrame, 
                business_df: pd.DataFrame, 
                predict_per_user: int = 10) -> Union[np.ndarray, pd.Series]:
        # Page rank could not predict stars.
        review_val_df['pred_stars'] = np.ones((len(review_val_df)))
        total_df = pd.Series()
        for i in user_df.user_id.unique():
            idx = self.train_df[self.train_df.user_id == i].user_id.cat.codes.max()
            # Cold start baseline
            if pd.isnull(idx):
                df = pd.Series([self.sorted[:predict_per_user].tolist()],index=[i])
                total_df = pd.concat([total_df,df])
                review_val_df.loc[review_val_df['user_id'] == i, 'pred_stars'] = review_val_df.loc[review_val_df['user_id'] == i, 'business_id'].apply(lambda business: self.business_median_stars.get(business, self.median_stars))
            else:
                # Find most similar users
                user_weights = nx.pagerank(self.G, personalization={idx:100000})
                ratings = self.train_df[['user_id','business_id','stars']].copy()

                # Add weights to calculate best business
                ratings['weight'] = self.train_df['user_codes'].apply(user_weights.get)
                ratings.loc[ratings['user_id'].cat.codes == idx, 'stars'] = 0

                ratings['ranking'] = (ratings['stars'] - 1.5) * ratings['weight']

                # Sort by weighted sum
                ratings_grouped = ratings.groupby('business_id').ranking.sum().sort_values()[::-1].reset_index()['business_id']
                df = pd.Series([ratings_grouped[:predict_per_user].to_list()], index=[i])

                total_df = pd.concat([total_df, df])

                # Stars count prediction
                ratings['stars_weighted'] = ratings['stars'] * ratings['weight']

                # get weighted average stars if weights are big enough, median otherwise
                stars_new = ratings.groupby('business_id').stars_weighted.sum() / (ratings.groupby('business_id').weight.sum() + 1e-6) * (ratings.groupby('business_id').weight.sum() > 1e-6) \
                    + self.median_stars * (ratings.groupby('business_id').weight.sum() < 1e-6)

                review_val_df.loc[review_val_df['user_id'] == i, 'pred_stars'] = review_val_df.loc[review_val_df['user_id'] == i, 'business_id'].apply(lambda business: stars_new.get(business, self.median_stars))

        return review_val_df['pred_stars'], total_df
