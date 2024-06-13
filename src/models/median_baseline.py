from functools import partial

import numpy as np
import pandas as pd

from src.models._base import BaseModel


class MedianBaselineModel(BaseModel):
    def __init__(self, target_col: str = 'stars', unique_stars: list = [1., 2., 3., 4., 5.,]) -> None:
        super().__init__(target_col, unique_stars)
        self.fitted = False
        self.review_train_df = None

    def fit(self, review_train_df: pd.DataFrame, user_df: pd.DataFrame, business_df: pd.DataFrame) -> None:
        self.business_stars_medians = review_train_df.groupby('business_id')['stars'].median()
        self.business_stars_medians_median = self.business_stars_medians.median()
        self.fitted = True
        self.users_prev_businesses = review_train_df.groupby('user_id').business_id.unique()
    
    def __get_top(self, user_id: str, business_stars: pd.Series, top_k:int=10):
        if user_id in self.users_prev_businesses:
            user_prev_businesses = self.users_prev_businesses.loc[user_id]
        else:
            user_prev_businesses = []

        business_stars_notvisited = []
        for business_id, business_stars in business_stars.items():
            if business_id not in user_prev_businesses:
                business_stars_notvisited.append(business_id)
            if len(business_stars_notvisited) >= top_k:
                break
        return business_stars_notvisited

    def predict(self, review_val_df: pd.DataFrame, user_df: pd.DataFrame, business_df: pd.DataFrame, predict_per_user: int = 10) -> np.ndarray | pd.Series:
        if not self.fitted:
            raise Exception("Fit function should be run before predict here!")
        
        predicted_stars = review_val_df['business_id'].apply(partial(self.business_stars_medians.get, default=self.business_stars_medians_median))

        val_users_df = user_df.loc[user_df['user_id'].isin(review_val_df['user_id'])].copy()

        business_stars = business_df['business_id'].apply(partial(self.business_stars_medians.get, default=self.business_stars_medians_median))
        business_stars.index = business_df['business_id'].values
        business_stars = business_stars.sort_values(ascending=False)

        top_suggestions = val_users_df['user_id'].apply(partial(self.__get_top, business_stars=business_stars, top_k=predict_per_user))
        top_suggestions.index = val_users_df['user_id'].values

        return predicted_stars, top_suggestions
