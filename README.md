# FairGenomicPCA

CS 599 Convex Optimization @ Oregon State December 2020.
This is an implementation of several dimensionality reduction methods to compare the performance of PCA-like algorithms on group and individual fairness objectives. The variants are: 
* Conventional PCA via eigendecomposition
* Frank-White algorithm for maximizing Nash social welfare [Tantipongpipat et al 2020](https://arxiv.org/abs/1902.11281)
* Gradient descent algorithm for Pareto-efficient reconstruction.[Kamani et al 2019](https://arxiv.org/abs/1911.04931)

I compared these algorithms on the Adult and Credit datasets and replicate experiments from [Price et al 2006](https://www.nature.com/articles/ng1847)

The central dependencies for this project include pandas, numpy, scipy, jax, scikit-learn, and cvxpy. I have included both a pip-freeze.txt and a conda list if you have issues with libraries. (I used conda)

To test, please run:

```python genome_trials.py```

This is a small test on only the 500 SNPs with maximum variance. If you would like to test more, you can increase the KEEP_TOP_SNPs constant.

Also run

```python adult_trials.py```

You will see similar output, but the Pareto-PCA method will run and never finish.
