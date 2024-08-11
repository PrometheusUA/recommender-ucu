import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm
from torch import optim
from torchvision import transforms
from typing import Optional

from src.models._base import BaseModel
from src.ranking.lambdamart import LambdaMART
from src.ranking.allrank.data.dataset_loading import LibSVMDataset, FixLength, ToTensor, fix_length_to_longest_slate, create_data_loaders
from src.ranking.allrank.models.model_utils import get_torch_device
from src.ranking.allrank.models.model import make_model
from src.ranking.allrank.models.losses import listNet
from src.ranking.allrank.training.train_utils import fit


class LambdaMARTModel(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.],
                 number_of_trees=40, max_tree_depth=2, save_path:Optional[str]=None):
        super().__init__(target_col, unique_stars)
        self.embeddings_index = {}
        self.restaurant_vectors = None
        self.user_vectors = None
        self.model = None
        self.number_of_trees = number_of_trees
        self.max_tree_depth = max_tree_depth
        self.embedding_model = SentenceTransformer("all-MiniLM-L12-v2", device=get_torch_device())
        self.save_path = save_path

    def obtain_user_vector(self, user_reviews_df: pd.DataFrame):
        mean_stars = user_reviews_df['stars'].mean()
        positive_mask = user_reviews_df['stars'] >= mean_stars

        vectors_list = []
        for id in user_reviews_df.index[positive_mask]:
            vectors_list.append(self.restaurant_vectors[user_reviews_df.loc[id, 'business_id']])

        return np.mean(np.vstack(vectors_list), axis=0)

    def fit(self, review_train_df, user_df, business_df):
        review_train_df['review_vector'] = self.embedding_model.encode(review_train_df['text'].tolist()).tolist()
        review_train_df['review_vector'] = review_train_df['review_vector'].apply(np.array)

        self.restaurant_vectors = review_train_df.groupby('business_id')['review_vector'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0), include_groups=False)
        self.user_vectors = review_train_df.groupby('user_id').apply(self.obtain_user_vector, include_groups=False)
        
        self.default_restaurant_vector = self.restaurant_vectors.mean()
        self.default_user_vector = self.user_vectors.mean()

        review_train_df['user_vector'] = review_train_df['user_id'].map(self.user_vectors)
        review_train_df['restaurant_vector'] = review_train_df['business_id'].map(self.restaurant_vectors)
        review_train_df['feature_vector'] = review_train_df[['user_vector', 'restaurant_vector']].apply(lambda row: np.concatenate((row['user_vector'], row['restaurant_vector'])), axis=1)
        review_train_df['user_id'] = review_train_df['user_id'].astype("category")

        self.model = LambdaMART(
            training_features=np.stack(review_train_df['feature_vector'].values),
            training_queries=review_train_df['user_id'].cat.codes.values,
            training_scores=review_train_df[self.target_col].values,
            number_of_trees=self.number_of_trees,
            tree_max_depth=self.max_tree_depth,
        )

        self.model.fit()

        if self.save_path:
            self.model.save(self.save_path)

        self.review_train_df = review_train_df
    
    def predict(self, review_val_df, user_df, business_df, predict_per_user=10, batch_size=64):
        top_recommendations = []

        unique_users = review_val_df['user_id'].unique()
        for i in tqdm(range(0, len(unique_users), batch_size), desc="User batch prediction"):
            batch_users = unique_users[i:i+batch_size]

            queries = []
            features = []
            user_index_map = []
            restaurant_ids = []

            for i, user_id in enumerate(batch_users):
                if user_id not in self.user_vectors.index:
                    user_vector = self.default_user_vector
                else:
                    user_vector = self.user_vectors[user_id]

                visited_restaurants = self.review_train_df[self.review_train_df['user_id'] == user_id]['business_id'].unique()
                unvisited_restaurant_vectors = self.restaurant_vectors[~self.restaurant_vectors.index.isin(visited_restaurants)]

                user_features = np.concatenate((np.repeat(user_vector.reshape((1, -1)), 
                                                        unvisited_restaurant_vectors.shape[0], axis=0),
                                                np.stack(unvisited_restaurant_vectors.to_numpy())), axis=1)

                features.append(user_features)
                queries.append(np.ones(user_features.shape[0]) * i)
                user_index_map.extend([user_id] * user_features.shape[0])
                restaurant_ids.extend(self.restaurant_vectors[~self.restaurant_vectors.index.isin(visited_restaurants)].index.tolist())

            queries = np.concatenate(queries)
            features = np.concatenate(features)

            test_similarities = self.model.predict(queries, features)

            results_df = pd.DataFrame({
                'user_id': user_index_map,
                'restaurant_id': restaurant_ids,
                'score': test_similarities
            })

            batch_recommendations = results_df.groupby('user_id').apply(lambda x: x.nlargest(predict_per_user, 'score')['restaurant_id'].tolist(), include_groups=False)
            top_recommendations.append(batch_recommendations)

        top_recommendations = pd.concat(top_recommendations)

        return None, top_recommendations


class NNRankModel(BaseModel):
    def __init__(self, target_col='stars', unique_stars=[1., 2., 3., 4., 5.], val_frac = 0.1,
                 slate_length=512, epochs=2, lr=1e-3, batch_size=32,
                 save_path='./models/nn_ranking.pkl'):
        super().__init__(target_col, unique_stars)
        self.embeddings_index = {}
        self.restaurant_vectors = None
        self.user_vectors = None
        self.model = None
        self.device = get_torch_device()
        self.val_frac = val_frac
        self.slate_length = slate_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.embedding_model = SentenceTransformer("all-MiniLM-L12-v2", device=self.device)
        self.save_path = save_path

    def fit(self, review_train_df, user_df, business_df):
        train_val_split_point = int((1 - self.val_frac) * len(review_train_df))
        train_reviews, val_reviews = review_train_df.iloc[:train_val_split_point].copy(), review_train_df.iloc[train_val_split_point:].copy()

        train_reviews['review_vector'] = self.embedding_model.encode(train_reviews['text'].tolist()).tolist()
        train_reviews['review_vector'] = train_reviews['review_vector'].apply(np.array)

        self.restaurant_vectors = train_reviews.groupby('business_id')['review_vector'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0), include_groups=False)
        self.user_vectors = train_reviews.groupby('user_id')['review_vector'].apply(lambda x: np.mean(np.vstack(x.tolist()), axis=0) if len(x) > 0 else pd.NA, include_groups=False)

        self.user_vectors = self.user_vectors.dropna()

        self.default_restaurant_vector = self.restaurant_vectors.mean()
        self.default_user_vector = self.user_vectors.mean()

        self.user_vectors = self.user_vectors.apply(list).apply(str)

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

        val_reviews['user_vector'] = val_reviews['user_vector'].apply(eval).apply(np.array)

        val_reviews['restaurant_vector'] = val_reviews['restaurant_vector'].apply(
            lambda x: self.default_restaurant_vector if not isinstance(x, np.ndarray) and pd.isna(x) else x
        )

        val_reviews['feature_vector'] = val_reviews[['user_vector', 'restaurant_vector']].apply(lambda row: np.concatenate((row['user_vector'], row['restaurant_vector'])), axis=1)
        val_reviews['user_id'] = val_reviews['user_id'].astype("category")

        self.user_vectors = self.user_vectors.apply(eval).apply(np.array)

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
            output_path=self.save_path,
            gradient_clipping_norm=1.0,
            early_stopping_patience=1,
        )

        self.review_train_df = review_train_df
    
    def predict(self, review_val_df, user_df, business_df, predict_per_user=10, batch_size=64):
        top_recommendations = []

        user_index_map = []
        restaurant_ids = []
        test_similarities = []

        unique_users = review_val_df['user_id'].unique()

        self.model.eval()

        for i, user_id in enumerate(unique_users):
            if user_id not in self.user_vectors.index:
                user_vector = self.default_user_vector
            else:
                user_vector = self.user_vectors[user_id]

            visited_restaurants = self.review_train_df[self.review_train_df['user_id'] == user_id]['business_id'].unique()
            unvisited_restaurant_vectors = self.restaurant_vectors[~self.restaurant_vectors.index.isin(visited_restaurants)]

            user_features = np.concatenate((np.repeat(user_vector.reshape((1, -1)), 
                                                    unvisited_restaurant_vectors.shape[0], axis=0),
                                            np.stack(unvisited_restaurant_vectors.to_numpy())), axis=1)

            user_index_map.extend([user_id] * user_features.shape[0])
            restaurant_ids.extend(self.restaurant_vectors[~self.restaurant_vectors.index.isin(visited_restaurants)].index.tolist())

            with torch.no_grad():
                similarities = self.model.score(torch.tensor(user_features, device=self.device).unsqueeze(0).float(), 
                                                torch.zeros(len(user_features), device=self.device).unsqueeze(0).bool(), 
                                                torch.ones(len(user_features), device=self.device).unsqueeze(0).long(),
                )

                similarities = similarities[0].cpu().detach().tolist()

            test_similarities.extend(similarities)

        results_df = pd.DataFrame({
            'user_id': user_index_map,
            'restaurant_id': restaurant_ids,
            'score': test_similarities
        })

        batch_recommendations = results_df.groupby('user_id').apply(lambda x: x.nlargest(predict_per_user, 'score')['restaurant_id'].tolist(), include_groups=False)
        top_recommendations.append(batch_recommendations)

        top_recommendations = pd.concat(top_recommendations)

        return None, top_recommendations
