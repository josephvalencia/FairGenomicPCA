# FairGenomicPCA

CS 599 Convex Optimization @ Oregon State December 2020.
This is an implementation of several dimensionality reduction methods to compare the performance of PCA-like algorithms on group and individual fairness objectives. The variants are: 
* Conventional PCA via eigendecomposition
* Frank-White algorithm for maximizing Nash social welfare [Tantipongpipat et al 2020](https://arxiv.org/abs/1902.11281)
* Gradient descent algorithm for Pareto-efficient reconstruction.[Kamani et al 2019](https://arxiv.org/abs/1911.04931)

I compared these algorithms on the Adult and Credit datasets and replicate experiments from [Price et al 2006](https://www.nature.com/articles/ng1847)

You can download plink 1.9 [here](https://www.cog-genomics.org/plink2)
