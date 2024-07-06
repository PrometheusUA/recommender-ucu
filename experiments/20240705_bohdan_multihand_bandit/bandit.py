import tqdm
import pandas as pd

import random
import numpy as np

class MultiArmedBandit:
    def __init__(self,models,business_df,review_df,user_df,N=100):
        self.models = models
        self.scores = [0] * len(models)
        self.business_df = business_df
        self.review_df = review_df
        self.N = N
        self.review_df['date'] = pd.to_datetime(self.review_df['date'])
        self.user_df = user_df
        self.models_score = [0] * len(self.models) # Score for bandit to decide how to sort models
        self.bandit_score = 0
    def train_models(self,date):
        '''
        Fit inital models
        '''
        review_train_df = self.review_df.loc[self.review_df['date'] <= date]
        user_train_df = self.user_df.loc[self.user_df['user_id'].isin(review_train_df['user_id'].unique())]
        business_train_df = self.business_df.loc[self.business_df['business_id'].isin(review_train_df['business_id'].unique())]

        
        for model in self.models:
            print('START FITTING A MODEL')
            model.fit(review_train_df, user_train_df, business_train_df)
            print("MODEL IS FITTED")
            
            
    def get_predictions(self,reviews,users,date):
        '''
        Get best fits for all models
        '''
        ranks =[]
        for model in self.models:
            res = model.predict(reviews,self.user_df,self.business_df,predict_per_user=self.N)[1]
            ranks.append(res)
        return ranks
    
    def get_bandit_arm(self):
        print("REWRITE THIS FUNCTION IN YOUR MAIN CLASS!!!!")
        return 1
    
    def recalculate_bandit_score(self,ranks,stars,business_id):
        try:
            rank =  ranks.index(business_id)
            
        except:
            return
        self.bandit_score += (stars-2.5)**3/((1+rank) ** 0.5)
        
    def recalculate_models_score(self,ranks,user_id,business_id,stars):
        for i in range(len(ranks)):
            rank = ranks[i][ranks[i].index == user_id].to_list()[0]
            try:
                rank =  rank.index(business_id)
            
            except:
                rank = -1
            if rank == -1:
                self.models_score[i] += 0
            else:
                self.models_score[i] += (stars-2.5)**3/((1+rank) ** 0.5)


    def simulate_data(self,users,date):
        reviews = self.review_df[self.review_df.user_id.isin(users)]
        reviews = reviews[reviews.date > date].sort_values('date')
        ranks = self.get_predictions(reviews,users,date) # As our models are static we could precalculate all ranks and now use bandit
        for review in tqdm.tqdm(reviews.iterrows()):
            business_id = review[1].business_id
            stars = review[1].stars
            user_id = review[1].user_id
            bandit_system_id = self.get_bandit_arm()
            rank = ranks[bandit_system_id][ranks[bandit_system_id].index == user_id].to_list()[0]
            self.recalculate_bandit_score(rank,stars,business_id)
            self.recalculate_models_score(ranks,user_id,business_id,stars)

        return ranks
    


class GreadyBandit(MultiArmedBandit):
    '''
    In the case of the Greedy approach, we will calculate the score according to the formula described at the beginning for each of the models.

    Next, with a probability of 1-eps, we will show the best of the models by the number of points scored.

    And also with the eps probability of one of the models with fewer scored points
    '''
    
    def __init__(self,models,business_df,review_df,user_df,N=100,eps=0.2):
        super().__init__(models,business_df,review_df,user_df,N)
        self.eps = eps
    def get_bandit_arm(self):
        best_score = np.argmax(np.array(self.models_score))
        if random.random() < self.eps:
            choices = [i for i in range(len(self.models_score)) if i != best_score]
            return random.choice(choices)
        else:
            return best_score
        



class ThompsonBandit(MultiArmedBandit):
    '''
    In the case of Thompson Sampling, we will use data such as whether a particular 
    institution has been recommended for a certain user.

    If our system says that among N institutions there is an institution visited by the guest,
    then we will add one to success, otherwise we will add one to Failure

    It doesn't matter rank of recomendation.

    '''
    def __init__(self,models,business_df,review_df,user_df,N=100):

        super().__init__(models,business_df,review_df,user_df,N)
        self.models_succ_score = []
        for i in range(len(self.models)):
            self.models_succ_score.append([0,0])
    def get_bandit_arm(self):
        samples = [np.random.beta(s+1, f+1) for s, f in self.models_succ_score]  # add 1 because can't pass 0

        # Pick the arm with highest sampled estimate
        best_arm = np.argmax(samples)
        return best_arm
    
    
    def recalculate_models_score(self,ranks,user_id,business_id,stars):
        for i in range(len(ranks)):
            rank = ranks[i][ranks[i].index == user_id].to_list()[0]
            try:
                rank =  rank.index(business_id)
                self.models_succ_score[i][0]+=1
            except:
                rank = -1
                self.models_succ_score[i][1]+=1
            
            if rank == -1:
                self.models_score[i] += 0
            else:
                self.models_score[i] += (stars-2.5)**3/((1+rank) ** 0.5)
