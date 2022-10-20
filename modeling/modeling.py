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

if __name__ == '__main__':
    pass
