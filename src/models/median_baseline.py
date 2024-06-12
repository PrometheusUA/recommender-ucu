from functools import partial

import numpy as np
import pandas as pd

from src.models._base import BaseModel


class MedianBaselineModel(BaseModel):
    def __init__(self, target_col: str = 'stars', unique_stars: list = [1., 2., 3., 4., 5.,]) -> None:
        super().__init__(target_col, unique_stars)
        self.fitted = False

    def fit(self, review_train_df: pd.DataFrame, user_df: pd.DataFrame, business_df: pd.DataFrame) -> None:
        self.business_stars_medians = review_train_df.groupby('business_id')['stars'].median()
        self.business_stars_medians_median = self.business_stars_medians.median()
        self.fitted = True
    
    def predict(self, review_val_df: pd.DataFrame, user_df: pd.DataFrame, business_df: pd.DataFrame) -> np.ndarray | pd.Series:
        if not self.fitted:
            raise Exception("Fit function should be run before predict here!")
        
        predicted_stars = review_val_df['business_id'].apply(partial(self.business_stars_medians.get, default=self.business_stars_medians_median))

        return predicted_stars
