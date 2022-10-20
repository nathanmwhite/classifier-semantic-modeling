# Classifier semantic modeling
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

__description__ = """This file contains the code for clustering on the model representations provided by the models."""

import argparse
import os

from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
# TODO: determine approach for silhouette analysis with KMeans
# TODO: determine whether to pursue support for Expectation Maximization via Gaussian Mixture Models
# TODO: determine whether to focus solely on hierarchical approaches
# TODO: determine whether NeighborNet is appropriate and how to support it

# TODO: code to determine which words to consider
def obtain_classifier_words(corpus):
    pass

# TODO: code to obtain embedding vectors for each word
def obtain_vectors(model):
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO: add util.py checking the argument values
    # TODO: determine whether there should be a n_clusters argument
    # TODO: determine which other hyperparams are necessary
    parser.add_argument('--language', type=str, default='Hmong')
    parser.add_argument('--model_type', type=str, default='Word2Vec')
    parser.add_argument('--method', type=str, default='K-Means')
    args = parser.parse_args()
    
    # TODO: call training on method
    
    # TODO: call extraction of groups from method object
    
    # TODO: evaluate against relevant Net (HmongNet, Chinese, Japanese WordNet versions)
