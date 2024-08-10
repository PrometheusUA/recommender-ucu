from typing import Union
import numpy as np
from src.models._base import BaseModel
import tqdm
from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel
import pandas as pd
class SequentialModel(BaseModel):
    def __init__(self, target_col: str = 'stars', unique_stars: list = [1., 2., 3., 4., 5.,]) -> None:
        super().__init__(target_col, unique_stars)
        self.fitted = False
        self.review_train_df = None
        self.user_ids_to_nums = {}
        self.business_ids_to_nums = {}
        self.user_nums_to_ids = {}
        self.business_nums_to_ids = {}
        
        
    def fit(self, 
            review_train_df: pd.DataFrame, 
            user_df: pd.DataFrame, 
            business_df: pd.DataFrame,
            sequence_lenght = 10) -> None:  # `user_df` and `business_df` can be different between fit and predict
        #user_df = user_df.reset_index(drop=True).reset_index()
        #business_df = business_df.reset_index(drop=True).reset_index()
        self.user_ids_to_nums = {i[1]['user_id']:i[1]['index']+1 for i in user_df.iterrows()}
        self.business_ids_to_nums = {i[1]['business_id']:i[1]['index']+1 for i in business_df.iterrows()}

        self.user_nums_to_ids = {i[1]['index']+1:i[1]['user_id'] for i in user_df.iterrows()}
        self.business_nums_to_ids = {i[1]['index']+1:i[1]['business_id'] for i in business_df.iterrows()}
        
        review_train_df['user_id_num'] = review_train_df['user_id'].apply(lambda x:self.user_ids_to_nums[x])
        review_train_df['business_id_num'] = review_train_df['business_id'].apply(lambda x:self.business_ids_to_nums[x])
        review_train_df['np_timestamp'] = pd.to_datetime(review_train_df['date']).values.astype('datetime64[ns]')
        self.review_train_df = review_train_df
        train  = Interactions(review_train_df['user_id_num'].values, review_train_df['business_id_num'].values,timestamps=review_train_df['np_timestamp'].values)
        train = train.to_sequence(max_sequence_length=sequence_lenght)
        self.max_sequence_length = sequence_lenght
        self.model = ImplicitSequenceModel(n_iter=10)
        self.model.fit(train,verbose=False)
        return train

    def predict(self, 
                review_val_df: pd.DataFrame, 
                user_df: pd.DataFrame, 
                business_df: pd.DataFrame, 
                predict_per_user: int = 10) -> Union[np.ndarray, pd.Series]:
        total_data = []
        total_df = pd.DataFrame()
        for i in user_df.index.unique():
            dfs = self.review_train_df[self.review_train_df.user_id_num == i]
            dfs = dfs.sort_values('date')
            data = dfs.business_id_num.values
            if len(data) == 0:
                data = [0] 
            total_data.append(data)
            res = self.model.predict(data)
            top_n_indices =np.argpartition(res, -predict_per_user)[-predict_per_user:]
            top_n_indices = [self.business_nums_to_ids[x] for x in top_n_indices[::-1]]
            try:
                df = pd.Series([top_n_indices], index=[self.user_nums_to_ids[i + 1] ],name='user_suggestions')
            except:
                df = pd.Series([top_n_indices], index=[user_df[user_df.index == i].user_id.to_list()[0]],name='user_suggestions')
                
            total_df = pd.concat([total_df, df])
            
        review_val_df['pred_stars'] = 4
        
        return review_val_df['pred_stars'], total_df
                        
