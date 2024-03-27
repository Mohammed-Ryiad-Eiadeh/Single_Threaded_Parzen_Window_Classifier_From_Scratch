# Parzen Window Algorithm:

This is a non-parametric lazy algorithm; hence, it does not include a training phase where a model learns. Yet, it directly estimates the likelihood or the probability distribution of the datapoints via the training samples. Therefore, it does not have an error rate curve like some other classifiers such as logistics regression where the curve in such approach is deduced from the training data during the training phase. Yet it has an advantage of being lazy algorithm such that it can be efficient when the data is large since it provides the decision boundary according to any data distribution, however, as long as the data becomes larger its’ cost increases.

Now let's delve into how it works mathematically: Suppose we have 2 classes `Class_1` and `Class_2`, where each has 4 samples. And we have 1 test unseen sample `X` we want to classify to either `Class_1` or to `Class_2` given the window size `h` or the bandwidth. We follow the following steps:

Calculate the density estimation that this test case refers to each class. And this can be mathematically given by:

```
Density(X|Class_i) = \frac{1}{N} \left( \sum_{j=1}^{N} K\left( \sum_{f=1}^{d} \sqrt{(p_f - q_f)^2}/h \right) \right)
```

where `N` is the total number of training samples belonging to `Class_i`, `d` is the number of features of the feature vector.

`K` is the Gaussian kernel which is captured by:

```
K(u) = 1/√(2π) e^(-u^2/2)
```

where `u = Σ_f^d √((p_f-q_f)^2)/h`

After calculating the likelihood, we use Bayesian decision theory to make the classification properly such that:

```
Class_i = {Class_1, if P(X|Class_1) P(Class_1) > P(X|Class_2) P(Class_2),
           Class_2, Otherwise}
```

And if we have more than two classes, we take the class that provides the maximum value of `P(X|Class_1) P(Class_1)`. That's it.

