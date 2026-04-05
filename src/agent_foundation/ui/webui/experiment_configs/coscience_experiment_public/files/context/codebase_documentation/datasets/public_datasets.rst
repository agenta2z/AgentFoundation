Public Datasets
===============

This page documents the public benchmark datasets used for evaluation and
experimentation in the Generative Recommenders codebase.

MovieLens-1M
------------

**Source**: GroupLens Research

**URL**: http://files.grouplens.org/datasets/movielens/ml-1m.zip

**Description**: 1 million ratings from 6,040 users on 3,706 movies.
Classic benchmark for recommendation systems research.

**Statistics**:

- Users: ~6,040
- Items: 3,706 (expected count in preprocessor)
- Ratings: ~1,000,000
- Rating Scale: 1-5 stars
- Timestamp Range: 2000-2003

**Use Case**: Quick experimentation and debugging due to small size.

MovieLens-20M
-------------

**Source**: GroupLens Research

**URL**: http://files.grouplens.org/datasets/movielens/ml-20m.zip

**Description**: 20 million ratings from 138,493 users on 26,744 movies.
Primary benchmark for comparing with state-of-the-art methods.

**Statistics**:

- Users: ~138,493
- Items: 26,744 (expected count in preprocessor)
- Ratings: ~20,000,000
- Rating Scale: 0.5-5.0 stars (half-star increments)
- Timestamp Range: 1995-2015

**Use Case**: Standard benchmark for academic papers, base for synthetic expansion.

**Reported Results (NDCG@10)**:

+-------------+--------+
| Model       | Score  |
+=============+========+
| HSTU-large  | 0.3813 |
+-------------+--------+
| SASRec      | 0.377  |
+-------------+--------+
| BERT4Rec    | 0.368  |
+-------------+--------+
| GRU4Rec     | 0.367  |
+-------------+--------+

Amazon Books
------------

**Source**: Stanford SNAP

**URL**: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv

**Description**: Book ratings from Amazon product reviews. Much sparser than
MovieLens, representing real-world recommendation challenges.

**Statistics**:

- Users: ~8,000,000
- Items: 695,762 (expected count in preprocessor)
- Ratings: ~22,000,000
- Density: Very sparse (~0.0004%)

**Use Case**: Testing model performance on sparse, large-scale data.

**Reported Results (NDCG@10)**:

+-------------+--------+
| Model       | Score  |
+=============+========+
| HSTU-large  | 0.0709 |
+-------------+--------+
| SASRec      | 0.054  |
+-------------+--------+
| GRU4Rec     | 0.048  |
+-------------+--------+

Data License and Citation
-------------------------

**MovieLens**:

The MovieLens datasets are provided by GroupLens Research under their terms of use.
Please cite::

    F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
    History and Context. ACM Transactions on Interactive Intelligent Systems
    (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
    DOI: http://dx.doi.org/10.1145/2827872

**Amazon Books**:

The Amazon dataset is provided by Stanford SNAP. Please cite::

    R. He, J. McAuley. Ups and Downs: Modeling the Visual Evolution of Fashion
    Trends with One-Class Collaborative Filtering. WWW, 2016.
