import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ._base import BaseModel


class RatingsDataset(Dataset):
    def __init__(self, stars, user_ids, business_ids):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.businesses = torch.tensor(business_ids, dtype=torch.long)
        self.ratings = torch.tensor(stars, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.businesses[idx], self.ratings[idx]

class NCFModel(nn.Module):
    def __init__(self, n_users, n_businesses, n_embed):
        super(NCFModel, self).__init__()
        self.user_embed = nn.Embedding(n_users, n_embed)
        self.business_embed = nn.Embedding(n_businesses, n_embed)
        
        self.fc1 = nn.Linear(n_embed * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, business):
        user_emb = self.user_embed(user)
        business_emb = self.business_embed(business)

        x = torch.cat([user_emb, business_emb], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        res = self.sigmoid(x).squeeze()
        return res
    
class NNColaborativeModel(BaseModel):
    def __init__(self, 
                 target_col: str = 'stars',
                 unique_stars: list = [1., 2., 3., 4., 5.],
                 n_embed: int = 20,
                 learning_rate: float = 0.01,
                 batch_size: int = 64,
                 epochs: int = 10) -> None:
        super().__init__(target_col, unique_stars)
        self.n_embed = n_embed
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.median_score = 0

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

        self.median_score = review_train_df['stars'].median()

        reviews_user_codes = review_train_df['user_id'].map(self.user_ids2codes)
        reviews_business_codes = review_train_df['business_id'].map(self.business_ids2codes)

        self.model = NCFModel(n_users, n_businesses, self.n_embed).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        dataset = RatingsDataset(review_train_df['stars'].values, reviews_user_codes.values, reviews_business_codes.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for users, businesses, ratings in dataloader:
                users, businesses, ratings = users.to(self.device), businesses.to(self.device), ratings.to(self.device)
                ratings = (ratings - 1) / 4 # normalize to [0; 1] range
                self.optimizer.zero_grad()
                outputs = self.model(users, businesses)
                loss = self.criterion(outputs, ratings)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * users.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            if verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}')

    def recommend_restaurants(self, user_id, top_n: int = 10):
        visited_restaurants = self.review_train_df[self.review_train_df['user_id'] == user_id]['business_id'].unique()
        unvisited_businesses = [business_id for business_id in self.business_ids2codes if business_id not in visited_restaurants]

        if user_id not in self.user_ids2codes or len(unvisited_businesses) == 0:
            return np.random.choice(np.array(list(self.business_ids2codes.keys())), top_n, replace=False).tolist()

        user_codes = torch.tensor([self.user_ids2codes[user_id]] * len(unvisited_businesses), device=self.device)
        business_codes = torch.tensor([self.business_ids2codes[business_id] for business_id in unvisited_businesses], device=self.device)

        scores = []
        with torch.no_grad():
            for batch_start in range(0, len(user_codes), self.batch_size):
                scores.append(self.model(user_codes[batch_start:batch_start+self.batch_size], business_codes[batch_start:batch_start+self.batch_size]).cpu().numpy())

        scores = np.concatenate(scores, axis=0)

        restaurant_scores = [(score * 4 + 1, business_id) for score, business_id in zip(scores, unvisited_businesses)]
        restaurant_scores.sort(reverse=True, key=lambda x: x[0])

        return [business_id for _, business_id in restaurant_scores[:top_n]]

    def predict(self, 
                review_val_df: pd.DataFrame, 
                user_df: pd.DataFrame, 
                business_df: pd.DataFrame, 
                predict_per_user: int = 10) -> tuple[pd.Series, pd.Series]:
        user_codes = torch.tensor([self.user_ids2codes.get(user_id, -1) for user_id in review_val_df['user_id']], device=self.device)
        business_codes = torch.tensor([self.business_ids2codes.get(business_id, -1) for business_id in review_val_df['business_id']], device=self.device)

        known_user_mask = user_codes >= 0
        known_business_mask = business_codes >= 0

        predictions = torch.full((len(user_codes),), self.median_score, device=self.device, dtype=torch.float32)

        known_user_codes = user_codes[known_user_mask & known_business_mask]
        known_business_codes = business_codes[known_user_mask & known_business_mask]

        all_preds = []
        if len(known_user_codes) > 0:
            with torch.no_grad():
                for i in range(0, len(known_user_codes), self.batch_size):
                    batch_user_codes = known_user_codes[i:i+self.batch_size]
                    batch_business_codes = known_business_codes[i:i+self.batch_size]
                    batch_preds = self.model(batch_user_codes, batch_business_codes)
                    all_preds.append(batch_preds)

        all_preds = torch.cat(all_preds)
        predictions[known_user_mask & known_business_mask] = all_preds * 4 + 1

        predictions = predictions.cpu().detach().numpy().tolist()

        user_suggestions = {}
        for user_id in user_df['user_id'].unique():
            user_suggestions[user_id] = self.recommend_restaurants(user_id, top_n=predict_per_user)

        return pd.Series(predictions, index=review_val_df.index), pd.Series(user_suggestions, name='suggestions')
