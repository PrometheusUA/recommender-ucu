# Factorization approaches

**by Andrii Shevtsov**

## Alternating least squares approach

Here, we solve alternating least squares problem until convergence of the quadratic loss with L2 penalty.

Interesting observation is that we need to solve this problem in a particular order for convergence: first we should update factors of users or items, then other factor, and only after that we should update biases of users and items in the same order. When we updated factors and biases of some element without updating factors of the other, there were convergence issues.

Also, user and business biases help to solve cold start problem for new users and businesses a bit.

## Funk SVD approach

Here, we solve an optimization problem with iterative gradient-based algorithm.

It's a typical Funk SVD based on SGD, but it requires gradient clipping, because sometimes exploding gradients problem arises.

## Neural network colaborative filtering approach

Here, we implemented illustrated sample from the lecture, where both `user_id` and `business_id` are embedded to feature spaces and then those vectors are concatenated and passed through deep neural network.

We used `PyTorch` framework for NN. The main issue was to write the batch implementation of predictions. But it was worth it: with BS 1024 it speed up forward from minutes to seconds.

## Results

We used 100k first samples short evaluation, because those methods are also pretty slow.

New models have hardly beaten the Median baseline model.
