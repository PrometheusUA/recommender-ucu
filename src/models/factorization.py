import pandas as pd
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

from ._base import BaseModel


class AlternatingLeastSquaresModel(BaseModel):
    def __init__(self, 
                target_col: str = 'stars',
                unique_stars: list = [1., 2., 3., 4., 5.],
                n_factors: int = 20,
                reguralization_param: float = 0.1,
                eps = 1e-5) -> None:
        super().__init__(target_col, unique_stars)
        self.n_factors = n_factors
        self.reguralization_param = reguralization_param
        self.eps = eps
        self.user_codes = None
        self.business_codes = None
        self.user_factors = None
        self.business_factors = None
        self.user_biases = None
        self.business_biases = None
        self.global_bias = None
        self.business_user_matrix_coo = None
        self.business_user_matrix = None
        self.review_train_df = None

    def loss(self):
        self.business_user_matrix_coo = self.business_user_matrix.tocoo()
        loss_val = 0
        for b, u, stars in zip(self.business_user_matrix_coo.row, self.business_user_matrix_coo.col, self.business_user_matrix_coo.data):
            loss_val += (stars - self.user_factors[u].T@self.business_factors[b] - self.global_bias - self.business_biases[b] - self.user_biases[u]) ** 2
        loss_val += self.reguralization_param * np.linalg.norm(self.business_factors, ord='fro') ** 2
        loss_val += self.reguralization_param * np.linalg.norm(self.user_factors, ord='fro') ** 2
        loss_val += self.reguralization_param * np.linalg.norm(self.user_biases, ord=2) ** 2
        loss_val += self.reguralization_param * np.linalg.norm(self.business_biases, ord=2) ** 2
        return loss_val

    def fit(self, 
            review_train_df: pd.DataFrame, 
            user_df: pd.DataFrame, 
            business_df: pd.DataFrame,
            verbose: bool = False) -> None:
        n_users = review_train_df['user_id'].nunique()
        n_businesses = review_train_df['business_id'].nunique()

        self.review_train_df = review_train_df

        user_ids = review_train_df['user_id'].unique()
        codes = range(len(user_ids))
        self.user_ids2codes = {user_id: code for user_id, code in zip(user_ids, codes)}
        self.codes2user_ids = {code: user_id for user_id, code in zip(user_ids, codes)}

        business_ids = review_train_df['business_id'].unique()
        codes = range(len(business_ids))
        self.business_ids2codes = {business_id: code for business_id, code in zip(business_ids, codes)}
        self.codes2business_ids = {code: business_id for business_id, code in zip(business_ids, codes)}

        reviews_user_codes = review_train_df['user_id'].map(self.user_ids2codes)
        reviews_business_codes = review_train_df['business_id'].map(self.business_ids2codes)

        self.business_user_matrix = sparse.csr_matrix((review_train_df['stars'], (reviews_business_codes, reviews_user_codes)), shape=(n_businesses, n_users))

        self.global_bias = review_train_df['stars'].mean()
        self.user_biases = np.zeros(n_users)
        self.business_biases = np.zeros(n_businesses)
        self.user_factors = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.business_factors = np.random.normal(scale=1./self.n_factors, size=(n_businesses, self.n_factors))

        delta_loss = np.inf
        loss = self.loss()
        if verbose:
            print("Loss calculated")
        iteration = 0

        while delta_loss > self.eps:
            # fixed businesses, update users factors
            for u in tqdm(range(n_users)) if verbose else range(n_users):
                businesses_ids = self.business_user_matrix[:, u].indices
                stars = self.business_user_matrix[:, u].data

                if len(businesses_ids) > 0:
                    A = self.business_factors[businesses_ids].T @ self.business_factors[businesses_ids] + self.reguralization_param * np.eye(self.n_factors)
                    b = (stars - self.global_bias - self.business_biases[businesses_ids] - self.user_biases[u]) @ self.business_factors[businesses_ids]
                    new_user_factors = np.linalg.solve(A, b)
                    new_user_factors[new_user_factors < 0] = 0
                    self.user_factors[u] = new_user_factors

            # fixed users, update businesses factors
            for bi in tqdm(range(n_businesses)) if verbose else range(n_businesses):
                user_ids = self.business_user_matrix[b].indices
                stars = self.business_user_matrix[b].data

                if len(user_ids) > 0:
                    A = self.user_factors[user_ids].T @ self.user_factors[user_ids] + self.reguralization_param * np.eye(self.n_factors)
                    b = (stars - self.global_bias - self.user_biases[user_ids] - self.business_biases[bi]) @ self.user_factors[user_ids]
                    new_business_factors = np.linalg.solve(A, b)
                    new_business_factors[new_business_factors < 0] = 0
                    self.business_factors[bi] = new_business_factors

            # fixed businesses, update users biases
            for u in tqdm(range(n_users)) if verbose else range(n_users):
                businesses_ids = self.business_user_matrix[:, u].indices
                stars = self.business_user_matrix[:, u].data

                if len(businesses_ids) > 0:
                    self.user_biases[u] = (stars - self.global_bias - self.business_factors[businesses_ids] @ self.user_factors[u] - self.business_biases[businesses_ids]).sum() / (len(businesses_ids) + self.reguralization_param)

            # fixed users, update businesses biases
            for bi in tqdm(range(n_businesses)) if verbose else range(n_businesses):
                user_ids = self.business_user_matrix[b].indices
                stars = self.business_user_matrix[b].data

                if len(user_ids) > 0:
                    self.business_biases[bi] = (stars - self.global_bias - self.user_factors[user_ids] @ self.business_factors[bi] - self.user_biases[user_ids]).sum() / (len(user_ids) + self.reguralization_param)

            old_loss = loss
            loss = self.loss()
            delta_loss = old_loss - loss
            iteration += 1

            if verbose:
                print(f"{iteration}: {loss=:.6f}, {delta_loss=:.6f}")

    def predict_single_score(self, user_id, business_id):
        if user_id not in self.user_ids2codes and business_id not in self.business_ids2codes:
            return self.global_bias
        elif user_id not in self.user_ids2codes:
            return self.global_bias + self.business_biases[self.business_ids2codes[business_id]]
        elif business_id not in self.business_ids2codes:
            return self.global_bias + self.user_biases[self.user_ids2codes[user_id]]
        
        user_code = self.user_ids2codes[user_id]
        business_code = self.business_ids2codes[business_id]

        return self.global_bias + self.user_biases[user_code] + self.business_biases[business_code] + \
            self.user_factors[user_code] @ self.business_factors[business_code]

    def recommend_restaurants(self, user_id, top_n: int = 10):
        visited_restaurants = self.review_train_df[self.review_train_df['user_id'] == user_id]['business_id'].unique()

        restaurant_scores = [(self.predict_single_score(user_id, business_id), business_id) for business_id in self.business_ids2codes if business_id not in visited_restaurants]

        return list(reversed([restaurant for _, restaurant in sorted(restaurant_scores)[-top_n:]]))

    def predict(self, 
                review_val_df: pd.DataFrame, 
                user_df: pd.DataFrame, 
                business_df: pd.DataFrame, 
                predict_per_user: int = 10) -> np.ndarray | pd.Series:
        user_suggestions = user_df.groupby('user_id', observed=True)['user_id'].apply(lambda user_id: self.recommend_restaurants(user_id.iloc[0], top_n=predict_per_user))

        predictions = []
        for _, row in review_val_df[["user_id", "business_id"]].iterrows():
            predictions.append(self.predict_single_score(row['user_id'], row['business_id']))

        return pd.Series(predictions), user_suggestions


class FunkSVDModel(BaseModel):
    def __init__(self, 
                target_col: str = 'stars',
                unique_stars: list = [1., 2., 3., 4., 5.],
                n_factors: int = 20,
                reguralization_param: float = 0.1,
                learning_rate: float = 0.01,
                gradient_clip: float  = 5.0,
                n_epoch: int = 20) -> None:
        super().__init__(target_col, unique_stars)
        self.n_factors = n_factors
        self.reguralization_param = reguralization_param
        self.n_epoch = n_epoch
        self.gradient_clip = gradient_clip
        self.learning_rate = learning_rate
        self.user_codes = None
        self.business_codes = None
        self.user_factors = None
        self.business_factors = None
        self.user_biases = None
        self.business_biases = None
        self.global_bias = None
        self.business_user_matrix_coo = None
        self.business_user_matrix = None
        self.review_train_df = None

    def loss(self):
        loss_val = 0
        for b, u, stars in zip(self.business_user_matrix_coo.row, self.business_user_matrix_coo.col, self.business_user_matrix_coo.data):
            loss_val += abs(stars - self.user_factors[u].T@self.business_factors[b] - self.global_bias - self.business_biases[b] - self.user_biases[u])
        loss_val += self.reguralization_param * np.sum(np.abs(self.business_factors), axis=None)
        loss_val += self.reguralization_param * np.sum(np.abs(self.user_factors), axis=None)
        loss_val += self.reguralization_param * np.linalg.norm(self.user_biases, ord=1)
        loss_val += self.reguralization_param * np.linalg.norm(self.business_biases, ord=1)
        return loss_val
    
    def clip_grad(self, value):
        if value > self.gradient_clip:
            return self.gradient_clip
        if value < -self.gradient_clip:
            return -self.gradient_clip
        return value

    def fit(self, 
            review_train_df: pd.DataFrame, 
            user_df: pd.DataFrame, 
            business_df: pd.DataFrame,
            verbose: bool = False) -> None:
        n_users = review_train_df['user_id'].nunique()
        n_businesses = review_train_df['business_id'].nunique()

        self.review_train_df = review_train_df

        user_ids = review_train_df['user_id'].unique()
        codes = range(len(user_ids))
        self.user_ids2codes = {user_id: code for user_id, code in zip(user_ids, codes)}
        self.codes2user_ids = {code: user_id for user_id, code in zip(user_ids, codes)}

        business_ids = review_train_df['business_id'].unique()
        codes = range(len(business_ids))
        self.business_ids2codes = {business_id: code for business_id, code in zip(business_ids, codes)}
        self.codes2business_ids = {code: business_id for business_id, code in zip(business_ids, codes)}

        reviews_user_codes = review_train_df['user_id'].map(self.user_ids2codes)
        reviews_business_codes = review_train_df['business_id'].map(self.business_ids2codes)

        self.business_user_matrix = sparse.csr_matrix((review_train_df['stars'], (reviews_business_codes, reviews_user_codes)), shape=(n_businesses, n_users))
        self.business_user_matrix_coo = self.business_user_matrix.tocoo()

        self.global_bias = review_train_df['stars'].mean()
        self.user_biases = np.zeros(n_users)
        self.business_biases = np.zeros(n_businesses)
        self.user_factors = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.business_factors = np.random.normal(scale=1./self.n_factors, size=(n_businesses, self.n_factors))

        for feature in tqdm(range(self.n_factors)):
            if verbose:
                print(f"Feature {feature}")
            delta_loss = np.inf
            loss = self.loss()
            if verbose:
                print(f"Loss {loss:.3f}")

            for epoch in range(self.n_epoch):
                for b, u, stars in zip(self.business_user_matrix_coo.row, self.business_user_matrix_coo.col, self.business_user_matrix_coo.data):
                    predicted_stars = self.global_bias + self.user_biases[u] + self.business_biases[b] + \
                        self.user_factors[u] @ self.business_factors[b]
                    
                    error = stars - predicted_stars

                    duser_factor = self.clip_grad(error * self.business_factors[b, feature] - self.reguralization_param * self.user_factors[u, feature])
                    dbusiness_factor = self.clip_grad(error * self.user_factors[u, feature] - self.reguralization_param * self.business_factors[b, feature])

                    duser_bias = self.clip_grad(error - self.reguralization_param * self.user_biases[u])
                    dbusiness_bias = self.clip_grad(error - self.reguralization_param * self.business_biases[b])

                    self.user_factors[u, feature] += self.learning_rate * duser_factor
                    self.business_factors[b, feature] += self.learning_rate * dbusiness_factor
                    self.user_biases[u] += self.learning_rate * duser_bias
                    self.business_biases[b] += self.learning_rate * dbusiness_bias

                self.user_factors /= np.linalg.norm(self.user_factors, axis=1, keepdims=True) + 1e-9
                self.business_factors /= np.linalg.norm(self.business_factors, axis=1, keepdims=True) + 1e-9

                old_loss = loss
                loss = self.loss()
                delta_loss = old_loss - loss

                if verbose:
                    print(f"{epoch}: {loss=:.6f}, {delta_loss=:.6f}")

    def predict_single_score(self, user_id, business_id):
        if user_id not in self.user_ids2codes and business_id not in self.business_ids2codes:
            return self.global_bias
        elif user_id not in self.user_ids2codes:
            return self.global_bias + self.business_biases[self.business_ids2codes[business_id]]
        elif business_id not in self.business_ids2codes:
            return self.global_bias + self.user_biases[self.user_ids2codes[user_id]]
        
        user_code = self.user_ids2codes[user_id]
        business_code = self.business_ids2codes[business_id]

        return self.global_bias + self.user_biases[user_code] + self.business_biases[business_code] + \
            self.user_factors[user_code] @ self.business_factors[business_code]

    def recommend_restaurants(self, user_id, top_n: int = 10):
        visited_restaurants = self.review_train_df[self.review_train_df['user_id'] == user_id]['business_id'].unique()

        restaurant_scores = [(self.predict_single_score(user_id, business_id), business_id) for business_id in self.business_ids2codes if business_id not in visited_restaurants]

        return list(reversed([restaurant for _, restaurant in sorted(restaurant_scores)[-top_n:]]))

    def predict(self, 
                review_val_df: pd.DataFrame, 
                user_df: pd.DataFrame, 
                business_df: pd.DataFrame, 
                predict_per_user: int = 10) -> np.ndarray | pd.Series:
        user_suggestions = user_df.groupby('user_id', observed=True)['user_id'].apply(lambda user_id: self.recommend_restaurants(user_id.iloc[0], top_n=predict_per_user))

        predictions = []
        for _, row in review_val_df[["user_id", "business_id"]].iterrows():
            predictions.append(self.predict_single_score(row['user_id'], row['business_id']))

        return pd.Series(predictions), user_suggestions
