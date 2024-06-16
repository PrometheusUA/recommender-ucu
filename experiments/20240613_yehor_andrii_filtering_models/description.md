# Collaborative filtering approaches

**by Andrii Shevtsov and Yehor Hryha**

## Content-based approach

Here, we used content (reviews) to create embeddings for both users and restaurants and to recommend later.

To do this, we embedded reviews text with mean GloVe vector of its words.

To obtain restaurants vectors, we have averaged its reviews vectors, and to obtain users vectors, we have averaged only vectors of his restaurants that were rated better then his average.

To make a recommendation for a user, we have to obtain his vector, calculate cosine similarities with all the restaurants and then just take `n_recommendations` with the biggest similarities.

To predict stars for user-restaurant pairs, we take a cosine similarity of their vectors and normalize it. Then we adjust it to [1; 5] range.

When user or a restaurant were missing during training, we set their vector to the default: a mean of all the vectors of that type.

## User-user filtering approach

Here, we use user-business sparse matrix with stars to create a sparse matrix of user-user cosine similarities.

For each user in val, we find his similar users via those similarities, take `n_neighboours` of those with the biggest similarities, and use weighted average of their scores to assign to unvisited restaurants, ignoring unvisited restaurants by those neighbours. For restaurants not visited by neighbours we predict their median score. 

For users that are absent in the train, we also use a median baseline for predictions.

## Item-item (business-business) filtering approach

Here, we use business-user sparse matrix with stars, that is a transpose of the user-business matrix from the previous method.

For each restaurant that is visited by user, we find `n_neighboours` most similar restaurants that are not visited, and add this restaurant's scores multiplied by similarity score between current and it's unvisited neighbour.

So, we obtain a weighed average of similar visited restaurants scores as a result for each unvisited restaurant.

## Results

We used 100k first samples short evaluation, because those methods are pretty slow.

Median baseline is comparable with all the models by classification/regression metrics, but is ~10 times slower in ranking.

Still, runtime of a baseline is 8 seconds, while item-item model takes 2 minute 4 seconds to evaluate, user-user model takes 2 minutes 10 seconds and content-based model takes 25 minutes on this dataset size.

## What didn't worked out (errata)

(Or things that had worse performance then those discussed above)

### Content-based approach

- To create user vector in multiple ways: 
    - by subtracting worse-then-average vectors from better-then-average restaurant vectors;
    - by averaging user's reviews;
    - by weighing reviews vectors by their scores (stars) and their squares.
- To create restaurant vectors in multiple ways.
- To use sentence-transformers instead of GloVe (tooo slow).

### User-user filtering and Item-item filtering

- Normalization of scores in any form. We tried:
    - Global normalization (subtracting global mean)
    - Business normalization (subtacting business mean and dividing by business std (as well as without dividing))
    - User normalization (also subtracting mean/median, with or without dividing by std)
- Unweighed averaging
- Taking all neighbours
- Searching for clusters
