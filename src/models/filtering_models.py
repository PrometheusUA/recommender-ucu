import re
import string
import sys
sys.path.append('../../')

from os.path import join as pjoin

import pandas as pd
import numpy as np
import faiss
import nltk

from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import TruncatedSVD

from src.utils import read_json_df
from src.models._base import BaseModel

nltk.download('stopwords')

glove_file = '../../data/glove.6B.100d.txt'

class ContentBasedModel(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.], glove_file=glove_file):
        super().__init__(target_col, unique_stars)
        self.embeddings_index = {}
        self.restaurant_vectors = None
        self.user_vectors = None
        self.load_glove_vectors(glove_file)

    def preprocess_text(self, text):
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        stop_words = set(stopwords.words("english"))
        word_tokens = text.split()
        filtered_text = [word for word in word_tokens if word not in stop_words]
        filtered_text = ' '.join(filtered_text)
        return filtered_text

    def load_glove_vectors(self, glove_file):
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = vector

    def get_average_vector(self, text):
        words = text.split()

        word_vectors = [self.embeddings_index[word] for word in words if word in self.embeddings_index]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            some_key = list(self.embeddings_index.keys())[0]
            return np.zeros(self.embeddings_index[some_key].shape)

    def fit(self, review_train_df, user_df, business_df):
        review_train_df['cleaned_text'] = review_train_df['text'].apply(self.preprocess_text)
        review_train_df['review_vector'] = review_train_df['cleaned_text'].apply(lambda x: self.get_average_vector(x))
        review_train_df_filtered = review_train_df[review_train_df['review_vector'].apply(lambda x: not np.allclose(np.zeros(self.embeddings_index[list(self.embeddings_index.keys())[0]].shape), x))]
        
        self.restaurant_vectors = review_train_df_filtered.groupby('business_id')['review_vector'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0))
        self.user_vectors = review_train_df_filtered.groupby('user_id')['review_vector'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0))
        self.review_train_df = review_train_df_filtered 
        self.default_restaurant_vector = np.mean(np.vstack(self.restaurant_vectors.tolist()), axis=0)
        self.defaul_user_vector = np.mean(np.vstack(self.user_vectors.tolist()), axis=0)

    def recommend_restaurants(self, user_id, review_df, top_n=10):
        if user_id not in self.user_vectors.index:
            return []
        user_vector = self.user_vectors[user_id]
        visited_restaurants = self.review_train_df[self.review_train_df['user_id'] == user_id]['business_id'].unique()
        unvisited_restaurant_vectors = self.restaurant_vectors[~self.restaurant_vectors.index.isin(visited_restaurants)]

        similarities = cosine_similarity([user_vector], unvisited_restaurant_vectors.tolist())
        similarity_scores = list(zip(unvisited_restaurant_vectors.index, similarities[0]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        return [restaurant for restaurant, score in similarity_scores[:top_n]]

    def predict_stars(self, user_id, restaurant_id):
        if restaurant_id in self.restaurant_vectors:
            restaurant_vector = self.restaurant_vectors[restaurant_id]
        else:
            restaurant_vector = self.default_restaurant_vector
        if user_id in self.user_vectors:
            user_vector = self.user_vectors[user_id]
        else:
            user_vector = self.defaul_user_vector
        similarity_score = np.dot(user_vector, restaurant_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(restaurant_vector))
        predicted_rating = similarity_score * 5  # Scale similarity score to star rating
        return predicted_rating

    def predict(self, review_val_df, user_df, business_df, predict_per_user=10):
        predictions = []
        user_suggestions = user_df.groupby('user_id')['user_id'].apply(lambda user_id: self.recommend_restaurants(user_id.iloc[0], review_df))

        for _, row in review_val_df[["user_id", "business_id"]].iterrows():
            predictions.append(self.predict_stars(row['user_id'], row['business_id']))

        return pd.Series(predictions), user_suggestions
    
class UserUserCollaborativeFiltering(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.]):
        super().__init__(target_col, unique_stars)
        self.user_item_sparse_matrix = None
        self.reduced_matrix = None
        self.index = None
        self.user_ids_df = None
        self.business_ids_df = None

    def fit(self, review_train_df, user_df, business_df):
        # Merge dataframes to ensure we have matching records
        review_df_filtered = review_train_df.merge(user_df[['user_id']], on='user_id', how='inner')
        review_df_filtered = review_df_filtered.merge(business_df[['business_id']], on='business_id', how='inner')
        
        # Create user and business ID mappings
        user_ids = review_df_filtered['user_id'].astype('category').cat.codes
        business_ids = review_df_filtered['business_id'].astype('category').cat.codes
        stars = review_df_filtered[self.target_col]

        # Create the user-item matrix
        self.user_item_sparse_matrix = coo_matrix((stars, (user_ids, business_ids)))
        self.user_item_sparse_matrix = self.user_item_sparse_matrix.tocsr()
        
        # Create DataFrames for user and business IDs
        user_ids.name = "user_code"
        self.user_ids_df = pd.concat((user_ids, review_df_filtered['user_id']), axis=1)

        business_ids.name = "business_code"
        self.business_ids_df = pd.concat((business_ids, review_df_filtered['business_id']), axis=1)
        
        # Reduce dimensions using SVD
        svd = TruncatedSVD(n_components=100)
        self.reduced_matrix = svd.fit_transform(self.user_item_sparse_matrix)
        self.reduced_matrix = self.reduced_matrix.astype(np.float32)

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.reduced_matrix.shape[1])
        faiss.normalize_L2(self.reduced_matrix)
        self.index.add(self.reduced_matrix)

    def recommend_restaurants(self, user_id, review_df, n_recommendations=10, n_neighbours=100):
        user_code = self.user_ids_df.loc[self.user_ids_df['user_id'] == user_id, 'user_code'].iloc[0]
        sim_scores, sim_user_codes = self.index.search(self.reduced_matrix[user_code].reshape(1, -1), n_neighbours)

        sim_scores, sim_user_codes = sim_scores[0], sim_user_codes[0]
        sim_scores = sim_scores[sim_user_codes != user_code]
        sim_user_codes = sim_user_codes[sim_user_codes != user_code]

        similar_users_ratings = self.user_item_sparse_matrix[sim_user_codes].toarray()
        avg_similar_users_ratings = similar_users_ratings.mean(axis=0)

        visited_restaurants = review_df[review_df['user_id'] == user_id]['business_id'].unique()
        visited_business_codes = self.business_ids_df[self.business_ids_df['business_id'].isin(visited_restaurants)]['business_code'].values

        recommendations_codes = [code for code in np.argsort(avg_similar_users_ratings)[::-1] if code not in visited_business_codes][:n_recommendations]

        res = []
        for code in recommendations_codes:
            business_id = self.business_ids_df.loc[self.business_ids_df['business_code'] == code, 'business_id'].values[0]
            res.append(business_id)
        
        return res

    def predict_stars(self, user_id, review_df):
        user_code = self.user_ids_df.loc[self.user_ids_df['user_id'] == user_id, 'user_code'].iloc[0]
        sim_scores, sim_user_codes = self.index.search(self.reduced_matrix[user_code].reshape(1, -1), 100)

        sim_scores, sim_user_codes = sim_scores[0], sim_user_codes[0]
        sim_scores = sim_scores[sim_user_codes != user_code]
        sim_user_codes = sim_user_codes[sim_user_codes != user_code]

        similar_users_ratings = self.user_item_sparse_matrix[sim_user_codes].toarray()
        avg_similar_users_ratings = similar_users_ratings.mean(axis=0)

        recommendations = self.recommend_restaurants(user_id, review_df)
        predicted_ratings = []
        for business_id in recommendations:
            business_code = self.business_ids_df.loc[self.business_ids_df['business_id'] == business_id, 'business_code'].values[0]
            predicted_rating = avg_similar_users_ratings[business_code]
            predicted_ratings.append(predicted_rating)

        return recommendations, predicted_ratings

    def predict(self, review_val_df, user_df, business_df, predict_per_user=10):
        predictions = []
        user_suggestions = []

        for user_id in user_df['user_id'].unique():
            if user_id in self.user_ids_df['user_id'].values:
                recommendations, predicted_ratings = self.predict_stars(user_id, review_val_df)
                predictions.extend(predicted_ratings[:predict_per_user])
                user_suggestions.extend(recommendations[:predict_per_user])

        return pd.Series(predictions), pd.Series(user_suggestions)
    
class ItemItemCollaborativeFiltering(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.]):
        super().__init__(target_col, unique_stars)
        self.user_item_sparse_matrix = None
        self.item_user_sparse_matrix = None
        self.reduced_matrix = None
        self.index = None
        self.user_ids_df = None
        self.business_ids_df = None

    def fit(self, review_train_df, user_df, business_df):
        # Merge dataframes to ensure we have matching records
        review_df_filtered = review_train_df.merge(user_df[['user_id']], on='user_id', how='inner')
        review_df_filtered = review_df_filtered.merge(business_df[['business_id']], on='business_id', how='inner')
        
        # Create user and business ID mappings
        user_ids = review_df_filtered['user_id'].astype('category').cat.codes
        business_ids = review_df_filtered['business_id'].astype('category').cat.codes
        stars = review_df_filtered[self.target_col]

        # Create the user-item matrix
        self.user_item_sparse_matrix = coo_matrix((stars, (user_ids, business_ids)))
        self.user_item_sparse_matrix = self.user_item_sparse_matrix.tocsr()

        # Transpose to get item-user matrix
        self.item_user_sparse_matrix = self.user_item_sparse_matrix.T
        
        # Create DataFrames for user and business IDs
        user_ids.name = "user_code"
        self.user_ids_df = pd.concat((user_ids, review_df_filtered['user_id']), axis=1)

        business_ids.name = "business_code"
        self.business_ids_df = pd.concat((business_ids, review_df_filtered['business_id']), axis=1)
        
        # Reduce dimensions using SVD
        svd = TruncatedSVD(n_components=100)
        self.reduced_matrix = svd.fit_transform(self.item_user_sparse_matrix)
        self.reduced_matrix = self.reduced_matrix.astype(np.float32)

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.reduced_matrix.shape[1])
        faiss.normalize_L2(self.reduced_matrix)
        self.index.add(self.reduced_matrix)

    def recommend_restaurants(self, user_id, review_df, n_recommendations=5, n_neighbours=100):
        # Get items rated by the user
        user_code = self.user_ids_df.loc[self.user_ids_df['user_id'] == user_id, 'user_code'].values[0]
        user_ratings = self.user_item_sparse_matrix[user_code].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]

        # Find similar items for each rated item
        recommendations = pd.Series()
        for item in rated_items:
            sim_scores, sim_item_codes = self.index.search(self.reduced_matrix[item].reshape(1, -1), n_neighbours)
            sim_scores, sim_item_codes = sim_scores[0], sim_item_codes[0]
            sim_scores = sim_scores[sim_item_codes != item]
            sim_item_codes = sim_item_codes[sim_item_codes != item]
            similar_items = pd.Series(sim_scores, index=sim_item_codes)
            recommendations = recommendations.append(similar_items)

        # Average the similarity scores
        recommendations = recommendations.groupby(recommendations.index).mean()
        
        # Get visited restaurant codes by the user
        visited_restaurants = review_df[review_df['user_id'] == user_id]['business_id'].unique()
        visited_business_codes = self.business_ids_df[self.business_ids_df['business_id'].isin(visited_restaurants)]['business_code'].values
        
        # Filter out already visited restaurants
        recommendations = recommendations.drop(visited_business_codes, errors='ignore')
        recommendations = recommendations.sort_values(ascending=False).head(n_recommendations)

        # Convert back to business IDs
        res = []
        for code in recommendations.index:
            business_id = self.business_ids_df.loc[self.business_ids_df['business_code'] == code, 'business_id'].values[0]
            res.append(business_id)
        
        return res

    def predict_stars(self, user_id, review_df):
        recommendations = self.recommend_restaurants(user_id, review_df)
        predicted_ratings = []

        user_code = self.user_ids_df.loc[self.user_ids_df['user_id'] == user_id, 'user_code'].values[0]
        user_ratings = self.user_item_sparse_matrix[user_code].toarray().flatten()

        for business_id in recommendations:
            business_code = self.business_ids_df.loc[self.business_ids_df['business_id'] == business_id, 'business_code'].values[0]
            sim_scores, sim_item_codes = self.index.search(self.reduced_matrix[business_code].reshape(1, -1), 100)
            sim_scores, sim_item_codes = sim_scores[0], sim_item_codes[0]
            sim_scores = sim_scores[sim_item_codes != business_code]
            sim_item_codes = sim_item_codes[sim_item_codes != business_code]
            similar_items_ratings = user_ratings[sim_item_codes]
            predicted_rating = np.dot(sim_scores, similar_items_ratings) / (np.abs(sim_scores).sum() + 1e-8)
            predicted_ratings.append(predicted_rating)

        return recommendations, predicted_ratings

    def predict(self, review_val_df, user_df, business_df, predict_per_user=10):
        predictions = []
        user_suggestions = []

        for user_id in user_df['user_id'].unique():
            if user_id in self.user_ids_df['user_id'].values:
                recommendations, predicted_ratings = self.predict_stars(user_id, review_val_df)
                predictions.extend(predicted_ratings[:predict_per_user])
                user_suggestions.extend(recommendations[:predict_per_user])

        return pd.Series(predictions), pd.Series(user_suggestions)