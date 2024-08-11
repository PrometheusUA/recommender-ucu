from typing import Iterable, Union

import pandas as pd
import numpy as np

from tqdm.autonotebook import tqdm

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
                business_df: pd.DataFrame, 
                predict_per_user: int = 10) -> tuple[Union[np.ndarray, pd.Series], pd.Series]:
        print("WARNING! Predict function is not overrided!")
        return np.ones((len(review_val_df)))

    @staticmethod
    def __calculate_ndcg(review_df, user_suggestions, suggestions_len):
        def dcg(relevances):
            return np.sum([
                (2**rel - 1) / np.log2(idx + 2) 
                for idx, rel in enumerate(relevances)
            ])
        
        def ndcg_at_k(relevances, k):
            actual_dcg = dcg(relevances[:k])
            ideal_dcg = dcg(sorted(relevances, reverse=True)[:k])
            return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        ndcg_scores = []
        
        for user_id, suggestions in user_suggestions.items():
            actual_relevances = []
            
            for business_id in suggestions:
                row = review_df[(review_df['user_id'] == user_id) & (review_df['business_id'] == business_id)]
                if not row.empty:
                    actual_relevances.append(row.iloc[0]['stars'])
                else:
                    actual_relevances.append(0)  # or a default relevance score for unseen items
            
            ndcg_scores.append(ndcg_at_k(actual_relevances, suggestions_len))

        return np.mean(ndcg_scores)

    def __metrics(self,
                  review_df: pd.DataFrame,
                  predicted_stars: Union[np.array, pd.Series],
                  user_suggestions: pd.Series,
                  suggestions_len: int = 10,
                  return_ranking_only: bool=False):

        # ranking mean precision @1, mean precision @3, mean precision @ K, MAP
        relevant_business_reviews = review_df[review_df[self.target_col] >= 4]
        relevant_user_businesses = relevant_business_reviews.groupby('user_id', observed=True)['business_id'].apply(list)
        relevant_user_businesses.name = "relevant_items"

        user_suggestions.name = "user_suggestions"
        user_relevant_vs_suggested = pd.merge(relevant_user_businesses, user_suggestions, how='outer', left_index=True, right_index=True)

        average_precisions_at_k = []
        user_relevant_vs_suggested = user_relevant_vs_suggested.map(lambda x: [] if (isinstance(x, float) and pd.isna(x)) else x)
        for k in range(1, suggestions_len + 1):
            user_relevant_vs_suggested['user_suggestions_topk'] = user_relevant_vs_suggested['user_suggestions'].apply(lambda x: x[:k])
            user_relevant_vs_suggested['relevant_topk'] = user_relevant_vs_suggested.apply(lambda row: len(set(row["user_suggestions_topk"]) & set(row["relevant_items"])), axis=1)

            average_precision_at_k = user_relevant_vs_suggested['relevant_topk'].mean() / k
            average_precisions_at_k.append(average_precision_at_k)

        map_k = np.mean(average_precisions_at_k)

        ndcg = BaseModel.__calculate_ndcg(review_df, user_suggestions, suggestions_len)

        if return_ranking_only:
            return {
                "AP@1": average_precisions_at_k[0],
                "AP@3": average_precisions_at_k[2],
                "AP@K": average_precisions_at_k[-1],
                "MAP@K": map_k,
                "NDCG": ndcg,
            }

        # regression
        real_stars = review_df[self.target_col]
        
        rmse = np.sqrt(mean_squared_error(real_stars, predicted_stars))
        mae = mean_absolute_error(real_stars, predicted_stars)
        # classification
        unique_stars = np.array(self.unique_stars,dtype=np.float32)
        # In case predicted stars are "between" existing classes, we need to adjust them to valid stars
        predicted_stars_classes = unique_stars[np.abs(predicted_stars.values.reshape((-1, 1)) - unique_stars.reshape((1, -1))).argmin(axis=1)]
        
        accuracy = accuracy_score(real_stars, predicted_stars_classes)
        f1 = f1_score(real_stars, predicted_stars_classes, average='macro', zero_division=1.0)
        precision = precision_score(real_stars, predicted_stars_classes, average='macro', zero_division=1.0)
        recall = recall_score(real_stars, predicted_stars_classes, average='macro', zero_division=1.0)

        return {
            "rmse": rmse,
            "mae": mae,

            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,

            "AP@1": average_precisions_at_k[0],
            "AP@3": average_precisions_at_k[2],
            "AP@K": average_precisions_at_k[-1],
            "MAP@K": map_k,
            "NDCG": ndcg,
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
                 time_folds_count: int = 5,
                 predict_per_user: int = 10, 
                 ranking_only: bool = False,
                ) -> dict:
        sorted_reviews_time = review_df['date'].sort_values().reset_index(drop=True)
        if short_eval:
            time_folds_count = 1
        
        folds_metrics = []

        for tfold in tqdm(range(time_folds_count), desc="Evaluation fold"):
            if not short_eval:
                review_small_div = sorted_reviews_time.loc[min(int(len(review_df) * ((2 + tfold) / (1 + time_folds_count))), len(review_df) - 1)]
                review_df_small = review_df.loc[review_df['date'] < review_small_div]

                review_small_train_div = review_df_small['date'].sort_values().reset_index(drop=True).loc[int(len(review_df_small) * ((tfold + 1)/(tfold + 2)))]
            else:
                review_small_div = sorted_reviews_time.iloc[short_eval_train_samples]
                review_df_small = review_df.loc[review_df['date'] <= review_small_div]

                review_small_train_div = review_df_small['date'].sort_values().reset_index(drop=True).loc[int(short_eval_train_samples * (1 - short_eval_val_size))]

            review_val_df = review_df_small.loc[review_df['date'] > review_small_train_div].copy()
            review_train_df = review_df_small.loc[review_df['date'] <= review_small_train_div].copy()

            user_train_df = user_df.loc[user_df['user_id'].isin(review_train_df['user_id'].unique())].copy()
            business_train_df = business_df.loc[business_df['business_id'].isin(review_train_df['business_id'].unique())].copy()

            user_val_df = user_df.loc[user_df['user_id'].isin(review_val_df['user_id'].unique())].copy()
            business_val_df = business_df.loc[business_df['business_id'].isin(review_val_df['business_id'].unique())].copy()

            self.fit(review_train_df, user_train_df, business_train_df)
            predicted_stars, user_suggestions = self.predict(review_val_df, user_val_df, business_val_df, predict_per_user)

            metrics = self.__metrics(review_val_df, predicted_stars, user_suggestions, predict_per_user, return_ranking_only=ranking_only)

            folds_metrics.append(metrics)

        return BaseModel.__average_folds_metrics(folds_metrics)
