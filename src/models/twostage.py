from typing import Optional

import pandas as pd
import numpy as np
import torch
from torch import optim
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.models._base import BaseModel
from src.ranking.allrank.data.dataset_loading import LibSVMDataset, FixLength, ToTensor, fix_length_to_longest_slate, create_data_loaders
from src.ranking.allrank.models.model_utils import get_torch_device
from src.ranking.allrank.models.model import make_model
from src.ranking.allrank.models.losses import listNet
from src.ranking.allrank.training.train_utils import fit


class TwoStage_KNN_NNModel(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.], val_frac = 0.1, trained_nn_path: Optional[str] = None, 
                 save_nn_path: Optional[str] = None, epochs:int=5, lr: float = 1e-3, slate_length: int = 512, batch_size: int = 16):
        assert trained_nn_path is None or save_nn_path is None, "We should either used trained NN for ranking or fit and save a new one"

        super().__init__(target_col, unique_stars)
        self.embeddings_index = {}
        self.restaurant_vectors = None
        self.user_vectors = None
        self.val_frac = val_frac
        self.device = get_torch_device()
        self.embedding_model = SentenceTransformer("all-MiniLM-L12-v2", device=self.device)
        self.trained_nn_path = trained_nn_path
        self.save_nn_path = save_nn_path
        self.epochs = epochs
        self.slate_length = slate_length
        self.lr = lr
        self.batch_size = batch_size

    def obtain_user_vector(self, user_reviews_df: pd.DataFrame):
        mean_stars = user_reviews_df['stars'].mean()
        positive_mask = user_reviews_df['stars'] >= mean_stars

        vectors_list = []
        for id in user_reviews_df.index[positive_mask]:
            vectors_list.append(self.restaurant_vectors[user_reviews_df.loc[id, 'business_id']])
        
        return np.mean(np.vstack(vectors_list), axis=0)

    def fit(self, review_train_df, user_df, business_df):
        train_val_split_point = int((1 - self.val_frac) * len(review_train_df)) if not self.trained_nn_path else len(review_train_df)
        train_reviews, val_reviews = review_train_df.iloc[:train_val_split_point].copy(), review_train_df.iloc[train_val_split_point:].copy()

        train_reviews['review_vector'] = self.embedding_model.encode(train_reviews['text'].tolist()).tolist()
        train_reviews['review_vector'] = train_reviews['review_vector'].apply(np.array)

        self.restaurant_vectors = train_reviews.groupby('business_id')['review_vector'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0), include_groups=False)
        self.user_vectors = train_reviews.groupby('user_id').apply(self.obtain_user_vector)
        
        self.default_restaurant_vector = self.restaurant_vectors.mean()
        self.default_user_vector = self.user_vectors.mean()

        if not self.trained_nn_path:

            train_reviews['restaurant_vector'] = train_reviews['business_id'].map(self.restaurant_vectors)
            train_reviews['user_vector'] = train_reviews['user_id'].map(self.user_vectors)
            train_reviews['user_vector'] = train_reviews['user_vector'].apply(eval).apply(np.array)

            train_reviews['feature_vector'] = train_reviews[['user_vector', 'restaurant_vector']].apply(lambda row: np.concatenate((row['user_vector'], row['restaurant_vector'])), axis=1)
            train_reviews['user_id'] = train_reviews['user_id'].astype("category")

            val_reviews['user_vector'] = val_reviews['user_id'].map(self.user_vectors)
            val_reviews['restaurant_vector'] = val_reviews['business_id'].map(self.restaurant_vectors)
                    
            val_reviews['user_vector'] = val_reviews['user_vector'].apply(
                lambda x: str(list(self.default_user_vector)) if not isinstance(x, str) and pd.isna(x) else x
            )

            val_reviews['restaurant_vector'] = val_reviews['restaurant_vector'].apply(
                lambda x: self.default_restaurant_vector if not isinstance(x, np.ndarray) and pd.isna(x) else x
            )

            val_reviews['feature_vector'] = val_reviews[['user_vector', 'restaurant_vector']].apply(lambda row: np.concatenate((row['user_vector'], row['restaurant_vector'])), axis=1)

            train_ds = LibSVMDataset(np.stack(train_reviews['feature_vector'].values),
                        train_reviews['stars'].values,
                        train_reviews['user_id'].cat.codes.values,
                        transform=transforms.Compose([FixLength(self.slate_length), ToTensor()]))

            val_ds = LibSVMDataset(np.stack(val_reviews['feature_vector'].values),
                                    val_reviews['stars'].values,
                                    val_reviews['user_id'].cat.codes.values)
            
            val_ds.transform = fix_length_to_longest_slate(val_ds)

            self.n_features = train_ds.shape[-1]

            train_dl, val_dl = create_data_loaders(train_ds, val_ds, 1, self.batch_size)

            self.model = make_model(
                {'sizes': [512, 256], 'input_norm': False, 'activation': 'ReLU', 'dropout': 0.2},
                {'positional_encoding': {'strategy': "learned", 'max_indices': self.slate_length},
                    "N": 1, "d_ff": 256}, 
                {'d_output': 1, 'output_activation': 'Sigmoid'}, 
                n_features=self.n_features
            )
            self.model.to(self.device)

            optimizer = optim.AdamW(lr=self.lr, params=self.model.parameters())
            loss_func = listNet
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, 1e-6)  # ReduceLROnPlateau(optimizer, factor=0.3, patience=0)

            fit(
                epochs=self.epochs,
                model=self.model,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dl=train_dl,
                valid_dl=val_dl,
                config={
                    'metrics': {'ndcg': None, 'dcg': None, 'mrr': None},
                    'val_metric': 'ndcg'
                },
                device=self.device,
                output_path=self.save_nn_path,
                gradient_clipping_norm=1.0,
                early_stopping_patience=0
            )
        else:
            self.n_features = 384 * 2
            self.model = make_model(
                {'sizes': [512, 256], 'input_norm': False, 'activation': 'ReLU', 'dropout': 0.2},
                {'positional_encoding': {'strategy': "learned", 'max_indices': self.slate_length},
                    "N": 1, "d_ff": 256}, 
                {'d_output': 1, 'output_activation': 'Sigmoid'}, 
                n_features=self.n_features
            )
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.trained_nn_path))
            self.model.eval()
        
        self.review_train_df = review_train_df

    def recommend_restaurants(self, user_id, review_df, top_n=10):
        # stage 1, retrieval
        retrieving_count = top_n * 10

        if user_id not in self.user_vectors.index:
            user_vector = self.default_user_vector
        else:
            user_vector = self.user_vectors[user_id]
        visited_restaurants = self.review_train_df[self.review_train_df['user_id'] == user_id]['business_id'].unique()
        unvisited_restaurant_vectors = self.restaurant_vectors[~self.restaurant_vectors.index.isin(visited_restaurants)]

        similarities = cosine_similarity([user_vector], unvisited_restaurant_vectors.tolist())
        similarity_scores = list(zip(unvisited_restaurant_vectors.index, similarities[0]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        retrieved_business_ids = [restaurant for restaurant, score in similarity_scores[:retrieving_count]]

        # stage 2, sorting
        retrieved_businesses_vectors = unvisited_restaurant_vectors[retrieved_business_ids]

        really_retrieved_count = retrieved_businesses_vectors.shape[0]

        user_features = np.concatenate((np.repeat(user_vector.reshape((1, -1)), 
                                                    really_retrieved_count, axis=0),
                                            np.stack(retrieved_businesses_vectors.to_numpy())), axis=1)

        with torch.no_grad():
            retrieved_businesses_scores = self.model.score(torch.tensor(user_features, device=self.device).unsqueeze(0).float(), 
                                                            torch.zeros(really_retrieved_count, device=self.device).unsqueeze(0).bool(), 
                                                            torch.ones(really_retrieved_count, device=self.device).unsqueeze(0).long(),
            )

            retrieved_businesses_scores = retrieved_businesses_scores[0].cpu().detach().tolist()

        businesses_scores = list(zip(retrieved_business_ids, retrieved_businesses_scores))
        businesses_scores.sort(key=lambda x: x[1], reverse=True)

        return [restaurant for restaurant, score in businesses_scores[:top_n]]

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
        user_suggestions = user_df.groupby('user_id')['user_id'].progress_apply(lambda user_id: self.recommend_restaurants(user_id.iloc[0], review_val_df, top_n=predict_per_user))

        for _, row in review_val_df[["user_id", "business_id"]].iterrows():
            predictions.append(self.predict_stars(row['user_id'], row['business_id']))

        return pd.Series(predictions), user_suggestions