from typing import Iterable, Union

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score


class BaseModel:
    def __init__(self, 
                 target_col: str = 'stars',
                 unique_stars: list = [1., 2., 3., 4., 5.]) -> None:
        self.target_col = target_col
        self.unique_stars = unique_stars

    def fit(self, 
            review_train_df: pd.DataFrame, 
            user_df: pd.DataFrame, 
            business_df: pd.DataFrame) -> None:  # `user_df` and `business_df` can be different between fit and predict
        print("WARNING! Fit function is not overrided!")

    def predict(self, 
                review_val_df: pd.DataFrame, 
                user_df: pd.DataFrame, 
                business_df: pd.DataFrame) -> Union[np.ndarray, pd.Series]:
        print("WARNING! Predict function is not overrided!")
        return np.ones((len(review_val_df)))

    def __metrics(self,
                  real_stars: Union[np.array, pd.Series],
                  predicted_stars: Union[np.array, pd.Series]):
        # classification
        rmse = np.sqrt(mean_squared_error(real_stars, predicted_stars))
        mae = mean_absolute_error(real_stars, predicted_stars)
        # regression
        unique_stars = np.array(self.unique_stars,dtype=np.float32)
        # In case predicted stars are "between" existing classes, we need to adjust them to valid stars
        predicted_stars_classes = unique_stars[np.abs(predicted_stars.values.reshape((-1, 1)) - unique_stars.reshape((1, -1))).argmin(axis=1)]
        
        accuracy = accuracy_score(real_stars, predicted_stars_classes)
        f1 = f1_score(real_stars, predicted_stars_classes, average='macro', zero_division=1.0)
        precision = precision_score(real_stars, predicted_stars_classes, average='macro', zero_division=1.0)
        recall = recall_score(real_stars, predicted_stars_classes, average='macro', zero_division=1.0)

        # ranking
        ## TODO: add ranking metrics
        return {
            "rmse": rmse,
            "mae": mae,

            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    
    @staticmethod
    def __average_folds_metrics(metrics: Iterable[dict]) -> dict:
        mean_metrics = metrics[0].copy()
        for fold_metrics in metrics[1:]:
            for metric_name in fold_metrics:
                mean_metrics[metric_name] += fold_metrics[metric_name]
        
        for metric_name in mean_metrics:
            mean_metrics[metric_name] /= len(metrics)

        return mean_metrics

    def evaluate(self,
                 review_df: pd.DataFrame,
                 user_df: pd.DataFrame,
                 business_df: pd.DataFrame,
                 short_eval: bool = False,
                 short_eval_train_samples: int = 1000,
                 short_eval_val_size: float = 0.1,
                 time_folds_count: int = 5
                ) -> dict:
        sorted_reviews_time = review_df['date'].sort_values().reset_index(drop=True)
        if short_eval:
            time_folds_count = 1
        
        folds_metrics = []

        for tfold in range(time_folds_count):
            if not short_eval:
                review_small_div = sorted_reviews_time.loc[min(int(len(review_df) * ((2 + tfold) / (1 + time_folds_count))), len(review_df) - 1)]
                review_df_small = review_df.loc[review_df['date'] < review_small_div]

                review_small_train_div = review_df_small['date'].sort_values().reset_index(drop=True).loc[int(len(review_df_small) * ((tfold + 1)/(tfold + 2)))]
            else:
                review_small_div = sorted_reviews_time.iloc[-short_eval_train_samples]
                review_df_small = review_df.loc[review_df['date'] > review_small_div]

                review_small_train_div = review_df_small['date'].sort_values().reset_index(drop=True).loc[int(short_eval_train_samples * (1 - short_eval_val_size))]

            review_val_df = review_df_small.loc[review_df['date'] > review_small_train_div]
            review_train_df = review_df_small.loc[review_df['date'] <= review_small_train_div]

            user_train_df = user_df.loc[user_df['user_id'].isin(review_train_df['user_id'].unique())]
            business_train_df = business_df.loc[business_df['business_id'].isin(review_train_df['business_id'].unique())]

            user_val_df = user_df.loc[user_df['user_id'].isin(review_val_df['user_id'].unique())]
            business_val_df = business_df.loc[business_df['business_id'].isin(review_val_df['business_id'].unique())]

            self.fit(review_train_df, user_train_df, business_train_df)
            predicted_stars = self.predict(review_val_df, user_val_df, business_val_df)

            metrics = self.__metrics(review_val_df[self.target_col], predicted_stars)

            folds_metrics.append(metrics)
        
        return BaseModel.__average_folds_metrics(folds_metrics)
