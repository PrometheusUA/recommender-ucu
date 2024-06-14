### PageRank
PageRank is an algorithm built on graphs to determine the final state of the system after N steps that simulate user behavior.

In our case, the graph is built as follows: users are vertices, and edges are dependencies between users. 

Therefore, our main task is to determine the weight of each edge. To do this, we decided to look for the Euclidean distance between two vectors that represent the rating of the business by the respective user. Since greater distance means less similarity, we take the inverse value: 

$$edge[i][j] = maxdistance - distance[i][j]$$

Similarity functions such as cosine similarity can also be the weight of the edges, but we focused on the inverse distance functions.

Next, to determine recommendations, we run PageRank from the vertex corresponding to the user.

After launch, we will get the probability that our state will be completed in each of the vertices, this will be the weight of the given user.

As a final step, we look for a weighted score for each of the businesses and sort them


## $$ rank_{user_{id}, business_{id}} = \sum_{x\in user}weight_{user_{id}, x} * stars_{x, business_{id}}$$
