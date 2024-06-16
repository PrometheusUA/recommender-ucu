import re
import string
import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
import nltk

from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix

from src.models._base import BaseModel


nltk.download('stopwords')

GLOVE_FILE = '../../data/glove.6B.100d.txt'


class ContentBasedModel(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.], glove_file=GLOVE_FILE):
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

    def obtain_user_vector(self, user_reviews_df: pd.DataFrame):
        mean_stars = user_reviews_df['stars'].mean()
        positive_mask = user_reviews_df['stars'] >= mean_stars
        # negative_mask = user_reviews_df['stars'] < median_stars

        # if positive_mask.sum() + negative_mask.sum() == 0:
        #     return None

        vectors_list = []
        for id in user_reviews_df.index[positive_mask]:
            vectors_list.append(self.restaurant_vectors[user_reviews_df.loc[id, 'business_id']])

        # for id in user_reviews_df.index[negative_mask]:
        #     vectors_list.append(- (user_reviews_df.loc[id, 'stars'] - median_stars + 1e-3)**2 * self.restaurant_vectors[user_reviews_df.loc[id, 'business_id']])

        # positive_vectors = ((user_reviews_df.loc[positive_mask, 'stars'] - median_stars) * self.restaurant_vectors[user_reviews_df.loc[positive_mask, 'business_id']]).tolist()
        # negative_vectors = ((user_reviews_df.loc[negative_mask, 'stars'] - median_stars) * self.restaurant_vectors[user_reviews_df.loc[negative_mask, 'business_id']]).tolist()

        return np.mean(np.vstack(vectors_list), axis=0)

    def fit(self, review_train_df, user_df, business_df):
        review_train_df['cleaned_text'] = review_train_df['text'].apply(self.preprocess_text)
        review_train_df['review_vector'] = review_train_df['cleaned_text'].apply(lambda x: self.get_average_vector(x))
        review_train_df_filtered = review_train_df[review_train_df['review_vector'].apply(lambda x: not np.allclose(np.zeros(self.embeddings_index[list(self.embeddings_index.keys())[0]].shape), x))]
        
        self.review_train_df = review_train_df_filtered

        self.restaurant_vectors = review_train_df_filtered.groupby('business_id')['review_vector'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0))
        self.user_vectors = review_train_df_filtered.groupby('user_id').apply(self.obtain_user_vector)
        
        self.default_restaurant_vector = self.restaurant_vectors.mean()
        self.default_user_vector = self.user_vectors.mean()

        # self.user_vectors.loc[pd.isnull(self.user_vectors)] = self.user_vectors.loc[pd.isnull(self.user_vectors)].apply(lambda x: self.default_user_vector)

    def recommend_restaurants(self, user_id, review_df, top_n=10):
        if user_id not in self.user_vectors.index:
            user_vector = self.default_user_vector
        else:
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
            user_vector = self.default_user_vector
        similarity_score = np.dot(user_vector, restaurant_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(restaurant_vector))
        predicted_rating = 1 + similarity_score * 4  # Scale similarity score to star rating
        return predicted_rating

    def predict(self, review_val_df, user_df, business_df, predict_per_user=10):
        predictions = []
        user_suggestions = user_df.groupby('user_id')['user_id'].apply(lambda user_id: self.recommend_restaurants(user_id.iloc[0], review_val_df, top_n=predict_per_user))

        for _, row in review_val_df[["user_id", "business_id"]].iterrows():
            predictions.append(self.predict_stars(row['user_id'], row['business_id']))

        return pd.Series(predictions), user_suggestions


class UserUserCollaborativeFiltering(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.], n_neighbours=20):
        super().__init__(target_col, unique_stars)
        self.user_item_sparse_matrix = None
        self.reduced_matrix = None
        self.index = None
        self.user_ids_df = None
        self.business_ids_df = None
        self.n_neighbours = n_neighbours

    def fit(self, review_train_df, user_df, business_df):
        # Merge dataframes to ensure we have matching records
        review_df_filtered = review_train_df.groupby(['user_id', 'business_id']).agg({'stars':'last','date':'last'}).reset_index()
        review_df_filtered = review_df_filtered.merge(user_df[['user_id']], on='user_id', how='inner')
        review_df_filtered = review_df_filtered.merge(business_df[['business_id']], on='business_id', how='inner')
        
        # Create user and business ID mappings
        user_codes = review_df_filtered['user_id'].astype('category').cat.codes
        business_codes = review_df_filtered['business_id'].astype('category').cat.codes

        self.median_stars = review_df_filtered['stars'].median()

        stars = review_df_filtered['stars']

        # Create the user-item matrix
        self.user_item_sparse_matrix = coo_matrix((stars, (user_codes, business_codes))).tocsr()
        
        # Create DataFrames for user and business IDs
        user_codes.name = "user_code"
        self.user_ids_df = pd.concat((user_codes, review_df_filtered['user_id']), axis=1)

        business_codes.name = "business_code"
        self.business_ids_df = pd.concat((business_codes, review_df_filtered['business_id']), axis=1).drop_duplicates()

        self.business_medians = review_df_filtered.groupby('business_id')['stars'].median()
        
        self.user_similarity = cosine_similarity(self.user_item_sparse_matrix, dense_output=False)

        self.sorted = review_train_df.groupby('business_id').stars.mean().sort_values().reset_index()['business_id'].to_numpy()[::-1]

    def predict(self, review_val_df, user_df, business_df, predict_per_user=10):
        user_suggestions = pd.Series()
        predicted_stars = pd.Series(np.ones((len(review_val_df))) * 3, index=review_val_df.index)

        for user_id in user_df.user_id.unique():
            if user_id not in self.user_ids_df['user_id'].unique():
                user_suggestions = pd.concat([user_suggestions, pd.Series([self.sorted[:predict_per_user].tolist()], index=[user_id])]) 
                predicted_stars.loc[review_val_df['user_id'] == user_id] = review_val_df.loc[review_val_df['user_id'] == user_id, 'business_id'].apply(lambda business: self.business_medians.get(business, self.median_stars))
                continue

            user_code = self.user_ids_df.loc[self.user_ids_df['user_id'] == user_id, 'user_code'].iloc[0]
            user_similarities = np.squeeze(np.asarray(self.user_similarity[user_code].toarray()))
            user_similarities[user_code] = 0

            new_user_similarities = np.zeros_like(user_similarities)
            new_user_similarities[np.argsort(user_similarities)[::-1][:self.n_neighbours]] = user_similarities[np.argsort(user_similarities)[::-1][:self.n_neighbours]]

            user_similarities = new_user_similarities.reshape((-1, 1))

            business_scores = np.squeeze(np.asarray((self.user_item_sparse_matrix.multiply(user_similarities)).sum(axis=0) / ((self.user_item_sparse_matrix > 0).multiply(user_similarities).sum(axis=0) + 1e-6)))

            top_codes = np.argsort(business_scores)[::-1][:predict_per_user]

            top_ids = []

            for business_code in top_codes:
                business_id = self.business_ids_df.loc[self.business_ids_df['business_code'] == business_code, 'business_id'].iloc[0]
                top_ids.append(business_id)
            
            user_suggestions = pd.concat([user_suggestions, pd.Series([top_ids], index=[user_id])])

            business_stars_ser = pd.Series(business_scores, index=self.business_ids_df.sort_values('business_code')['business_id'])
            business_stars_ser.loc[business_stars_ser == 0] = self.business_medians.loc[business_stars_ser.index[business_stars_ser == 0]]

            predicted_stars.loc[review_val_df['user_id'] == user_id] = review_val_df.loc[review_val_df['user_id'] == user_id, 'business_id'].apply(lambda business: business_stars_ser.get(business, self.median_stars))

        return predicted_stars, user_suggestions

    
class ItemItemCollaborativeFiltering(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.], n_neighbours=20):
        super().__init__(target_col, unique_stars)
        self.user_item_sparse_matrix = None
        self.reduced_matrix = None
        self.index = None
        self.user_ids_df = None
        self.business_ids_df = None
        self.n_neighbours = n_neighbours

    def fit(self, review_train_df, user_df, business_df):
        # Merge dataframes to ensure we have matching records
        review_df_filtered = review_train_df.groupby(['user_id', 'business_id']).agg({'stars':'last','date':'last'}).reset_index()
        review_df_filtered = review_df_filtered.merge(user_df[['user_id']], on='user_id', how='inner')
        review_df_filtered = review_df_filtered.merge(business_df[['business_id']], on='business_id', how='inner')
        
        # Create user and business ID mappings
        user_codes = review_df_filtered['user_id'].astype('category').cat.codes
        business_codes = review_df_filtered['business_id'].astype('category').cat.codes

        self.median_stars = review_df_filtered['stars'].median()

        stars = review_df_filtered['stars']

        # Create the user-item matrix
        self.item_user_sparse_matrix = coo_matrix((stars, (business_codes, user_codes))).tocsr()
        
        # Create DataFrames for user and business IDs
        user_codes.name = "user_code"
        self.user_ids_df = pd.concat((user_codes, review_df_filtered['user_id']), axis=1)

        business_codes.name = "business_code"
        self.business_ids_df = pd.concat((business_codes, review_df_filtered['business_id']), axis=1).drop_duplicates()

        self.business_medians = review_df_filtered.groupby('business_id')['stars'].median()
        
        self.business_similarity = cosine_similarity(self.item_user_sparse_matrix, dense_output=False)

        self.sorted = review_train_df.groupby('business_id').stars.mean().sort_values().reset_index()['business_id'].to_numpy()[::-1]

    def predict(self, review_val_df, user_df, business_df, predict_per_user=10):
        user_suggestions = pd.Series()
        predicted_stars = pd.Series(np.ones((len(review_val_df))) * 3, index=review_val_df.index)

        for user_id in user_df.user_id.unique():
            if user_id not in self.user_ids_df['user_id'].unique():
                user_suggestions = pd.concat([user_suggestions, pd.Series([self.sorted[:predict_per_user].tolist()], index=[user_id])]) 
                predicted_stars.loc[review_val_df['user_id'] == user_id] = review_val_df.loc[review_val_df['user_id'] == user_id, 'business_id'].apply(lambda business: self.business_medians.get(business, self.median_stars))
                continue

            user_code = self.user_ids_df.loc[self.user_ids_df['user_id'] == user_id, 'user_code'].iloc[0]
            user_ratings = self.item_user_sparse_matrix[:, user_code].toarray().flatten()

            business_scores = np.zeros(self.item_user_sparse_matrix.shape[0])
            business_weights = np.zeros(self.item_user_sparse_matrix.shape[0])

            for business_code in np.where(user_ratings > 0)[0]:
                businesses_similarity = self.business_similarity[business_code].toarray().flatten()
                businesses_similarity[business_code] = 0
                new_businesses_similarity = np.zeros_like(businesses_similarity)
                top_similar_indices = np.argsort(businesses_similarity)[::-1][:self.n_neighbours]
                new_businesses_similarity[top_similar_indices] = businesses_similarity[top_similar_indices]
                business_scores += new_businesses_similarity * user_ratings[business_code]
                business_weights += new_businesses_similarity

            business_scores[business_weights > 0] /= business_weights[business_weights > 0]

            top_codes = np.argsort(business_scores)[::-1][:predict_per_user]

            top_ids = []

            for business_code in top_codes:
                business_id = self.business_ids_df.loc[self.business_ids_df['business_code'] == business_code, 'business_id'].iloc[0]
                top_ids.append(business_id)
            
            user_suggestions = pd.concat([user_suggestions, pd.Series([top_ids], index=[user_id])])

            business_stars_ser = pd.Series(business_scores, index=self.business_ids_df.sort_values('business_code')['business_id'])
            business_stars_ser.loc[business_stars_ser == 0] = self.business_medians.loc[business_stars_ser.index[business_stars_ser == 0]]

            predicted_stars.loc[review_val_df['user_id'] == user_id] = review_val_df.loc[review_val_df['user_id'] == user_id, 'business_id'].apply(lambda business: business_stars_ser.get(business, self.median_stars))

        return predicted_stars, user_suggestions
