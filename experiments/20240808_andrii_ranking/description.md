# Ranking techniques

**by Andrii Shevtsov**

## General approcach

To use ranking techniques, we need to have feature vectors for items (restaurants) and users. We use content-based approach for this, using sentence transformer's representations of reviews to encode them. Then, we use the mean vector of all restaurant's reviews as its feature vector, and the mean vector of all the reviews person have written with more (or equal) stars than user have in his median-starred review.

Then, for comparison to use information about a user, we obtained a feature vector for user visiting a restaurant as a concatenation of restaurant vector and a user vector. Hopefuly, the algorithm should extrapolate well for new users with vectors counted in this way.

To evaluate the model, we have added an option to evaluate models with only ranking metrics, as all others have little sense to look after in our case. It is a `ranking_only` boolean parameter in `BaseModel.evaluate` method.

Also, we have added a Normalized discounted cumulative gain as a new metric to ranking metrics, it helps us to better distinguish between ranking approaches.

## LambdaMART

Here, we use gradient boosted decision trees with LambdaRank's cost function. It allows to directly optimize ranking metric by using gradients that represent an importance of swapping between two elements in the whole list. Therefore it is computed based on the effect of a swap on the ranking metric, and is directly optimized via gradient boosting.

To implement LambdaMART, we have refined an implementation of Ashish Agrawal and others \[[GitHub](https://github.com/lezzago/LambdaMart)\]. It trains fast, however testing it on large datasets was constantly causing performance issues.

## Neural Network for ranking

There are many ways to use neural networks for ranking. We followed a way covered in the allRank: Learning to Rank in PyTorch \[[GitHub](https://github.com/allegro/allRank)\]. Still, a lot of changes were needed to make it workable for our task.

The best option was to use ListNet loss and a fully connected model with one transformer encoder block that treats all the features of restaurants in query as one sequence. There was an option to use a much bigger transformer model, but it required many more data samples to actually learn patterns.

**ListNet** loss is based on a cross-entropy. It uses softmax to calculate a probability of each item in the list to be sorted as a top-1, and tries to make this distribution for a predicted values as close to the distribution for true values as possible (again, via cross-entropy formula). It is a list-level loss, not a pairwise like lambdas in LambdaMART.

## Results

We will compare only those two models as they are not intended to be used in this way. They both here are fitted on a 10k samples (reviews) and have 2k in a test set.

| **Model**  | **AP@1**| **AP@3**| **AP@10**| **MAP@10** | **NDCG**  |
| ---------- | ------: | ------: | -------: | ---------: | --------: |
| LambdaMART | 0       | 0.0009  |  0.0003  | 0.0005     | 0.0017    |
| Neural net | 0       | 0       |  0.0045  | 0.005      | 0.019     |
