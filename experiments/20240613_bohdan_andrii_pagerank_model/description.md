# PageRank

**by Bohdan Vey and Andrii Shevtsov**

## Algorithm description

PageRank is an algorithm built on graphs to determine the final state of the system after N steps that simulate user behavior.

In our case, the graph is built as follows: users are vertices, and edges are dependencies between users. 

Therefore, our main task is to determine the weight of each edge. To do this, we decided to look for the cosine similarity between two vectors that represent the rating of the business by the respective user. Sinse bigger similarity means bigger users correlation, we take it as a weight of the edge: 

$$edge[i][j] = similarity[i][j]$$

Cosine similarity allows to perform calculations in sparse matrices, which significally increases calculations speed.

Next, to determine recommendations, we run PageRank from the vertex corresponding to the user.

After launch, we will get the probability that our state will be completed in each of the vertices, this will be the weight of the given user.

As a final step, we look for a weighted score for each of the businesses and sort them:

$$rank_{user_{id}, business_{id}} = \sum_{x\in user}weight_{user_{id}, x} * stars_{x, business_{id}}$$

Top $N$ ranks are taken as a recommendation.

And set stars according to the weighted average:

$$stars_{user_{id}, business_{id}} = \frac{\sum_{x\in user}weight_{user_{id}, x} * stars_{x, business_{id}}}{\sum_{x\in user}weight_{user_{id}, x}}$$

## Results

The algorithm has similar classification/regression metrics with Median baseline, but it has significantly higher ranking metrics (2x to 10x improvement). Still, the algorithm is quite slow and is impractical for real use cases (For 10k users, it runs for 25 minutes).
