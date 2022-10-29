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

import xml.etree.ElementTree as ET

from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
# TODO: determine approach for silhouette analysis with KMeans
# TODO: determine whether to pursue support for Expectation Maximization via Gaussian Mixture Models
# TODO: determine whether to focus solely on hierarchical approaches
# TODO: determine whether NeighborNet is appropriate and how to support it

from .util import get_model, get_method_object
# TODO: get_method_object should contain objects wrapping external libraries to provide common interface

# TODO: code to determine which words to consider
# TODO: will need to develop polysyllabic word support for Hmong and Chinese
# TODO: this should essentially be able to crawl over the WordNet versions for each language
# HmongNet directly supports categories, others may not do so
# TODO: review best placement of the following dictionary
HMONG_CLF = {'tus': ['person', 'one_dimensional_object', 'animal', 'wild_animal', 'domestic_animal', 'social_role',\
                     'family_relation', 'consanguineal_relation', 'affinal_relation', 'ethnicity', 'symbol']}

def obtain_classifier_words(corpus, nom_clf, prefix=None):
    if corpus == 'HmongNet':
        classifier_words = []
        xml_data = ET.parse(prefix + 'hmong_ontology.xml')
        root = xml_data.get_root()
        # root > word_set > lemma > form > sense > category -- match category, then get form
        word_set = root[1]
        for lemma in word_set:
            for child in lemma:
                if child.tag == 'form':
                    form = child.text
                if child.tag == 'sense':
                    for c in child:
                        if c.tag == 'category':
                            if c.text in HMONG_CLF[nom_clf]:
                                classifier_words.append(form)
   return clsasifier_words

# TODO: code to obtain embedding vectors for each word
#  should return dictionary of word : index relations and
#  vectors from the model
# TODO: this should obtain the vectors from the models
def obtain_vectors(model, vocab):
    pass

# TODO: define 

if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO: add util.py checking the argument values
    # TODO: determine whether there should be a n_clusters argument
    # TODO: determine which other hyperparams are necessary
    parser.add_argument('--language', type=str, default='Hmong')
    parser.add_argument('--nom_clf', type=str, default='lub')
    parser.add_argument('--corpus', type=str, default='sch')
    parser.add_argument('--model_type', type=str, default='Word2Vec')
    parser.add_argument('--cluster_method', type=str, default='K-Means')
    args = parser.parse_args()
    
    vocab = obtain_classifier_words(args.corpus, args.nom_clf)
    
    indices, vectors = obtain_vectors(model, vocab)
    
    c_method = get_method_object(args.cluster_method)
    # TODO: determine what hyperparameters need to be present for train call
    c_method.train(vectors)
    
    # TODO: call extraction of groups from method object
    groups = c_method.get_groups()
    
    # TODO: evaluate against relevant Net (HmongNet, Chinese, Japanese WordNet versions)
