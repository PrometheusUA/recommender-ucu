# 2-Stage recommendation pipeline

**by Andrii Shevtsov**

## Approach

We decided to use very simple (and very fast) content-based filtering approach as a first step, followed by a NN-based ranking. As in the ranking approach, we used feature vectors based on sentence transformers (MiniLM) representation of reviews text. We have taken mean of restaurant reviews' vectors for restaurants, and mean of those that obtained score bigger than a median for user as his vector (to search for restaurants that are closer to those he liked). The top (n*10) restaurants by similarity are then sorted with a NN-based ranking algorithm.

## Benefits and drawbacks of the approach

Benefits of the approach include:
- High computation speed. Ranking is slow on big datasets, while filtering is fast. So, filtering works on a large dataset and returns a small list for ranking to work with, which is quite fast in total.
- Adequate precision. Filtering isn't very accurate. So, filtering possibly returns many OK variants that are then polished by ranking algorithm in an adequate time span, resulting in better precision than just returning first filtered samples.
- Modularity, meaning we can swap each step without tackling the other, and everything will work fine. Also, we can train and tune filtering to be focused on recall, and ranking to be focused on precision, resulting in possibly much better model, with a help of modularity.

While drawbacks are:
- Both models are fitted and tuned independently, so together they can work (especially ranking that should rank pretty good pre-chosen candidates).
- If the first model loses great candidate, it's lost for ranking too, even though ranking could have sent it to the top. Also, we don't have a precise way to choose a filtration size, meaning we should somehow trade off between ranking and filtering.

## Results

Classification and regression metrics here are based on filtering, so are not actually quite high, while all the ranking metrics are about 10 times higher than for a baseline, and a bit better than for just filtering approach.
