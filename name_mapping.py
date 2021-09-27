#!/usr/bin/env python3

"""
Script to map company names to their common entity. 

author: Simon Berlendis
date: 09/2021
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
import string
import nltk
from cleanco import basename, prepare_terms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster, leaders, leaves_list
from difflib import SequenceMatcher
from collections import Counter

def data_preprocessing(data):
    """
    Pre-process the data : decore, lower, remove special characters and defined stop-words.
    
    Parameters:
    data: pandas serie object containing the list of names  
    
    Returns:
    data_prep: pandas serie object with cleaned names
    """
    
    # encoding and lowering
    data_prep = data.str.normalize('NFKD').str.encode("ascii",'ignore').str.decode("utf-8","ignore")
    data_prep = data_prep.str.lower()

    # Remove organization type sequences (e.g. inc, llp)
    terms = prepare_terms()
    data_prep = data_prep.apply(lambda x: basename(x, terms)) 
    
    # remove special characters
    punctuation_custom = string.punctuation.replace('&','') # !"#$%'()*+,-./:;<=>?@[\]^_`{|}~
    data_prep = data_prep.apply(lambda x: re.sub('[%s]'%re.escape(punctuation_custom), '' , x)) ## ponctuation
    data_prep = data_prep.apply(lambda x: re.sub('  ', ' ' , x)) ## remove double space

    # Remove stop words
    # stop_words = ["gmbh", "& co", "co", "kg", "cokg", "ltd", "limited", "sl", "inc", "sa", "sarl", "sas", 
    #               "llc", "dba"]
    stop_words = ["cokg", "co", "gmbh", "gmbh&cokg", "association", "autohaus", "services", "group", "france", 
                  "technologies", "systems", "solutions", "consulting", "advanced", "asociacion", "engineering",
                  "akademie", "technology", "service", "international", "hotel", "europe", "industrie", "software",
                  "deutschland", "management", "agence"]
    data_prep = data_prep.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    return data_prep


def get_distance_matrix(data, ngram_range=(3,3)):
    """
    Vectorize the data using the N-gram and TF-ITF approach and compute the distance matrix.
    
    Parameters:
    data: pandas serie object containing the list of cleaned names  
    ngram_range: tuple defining the range for the n-gram vectorization (default (3,3))
    
    Returns:
    distance_matrix: scipy sparse matrix (csr_matrix) containaing the distance score between all names 
    """
    
    # Perform vectorization
    vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=ngram_range)
    data_vectorized = vectorizer.fit_transform(data)
    
    # Compute distance scores using cosine distance 
    distance_matrix = cosine_distances(data_vectorized)
    
    return distance_matrix


def get_linkage_matrix(distance_matrix, linkage='complete'):
    """
    Perform the hierarchical clustering using the distance matrix and build the resulting linkage matrix
    
    Parameters:
    distance_matrix: scipy sparse matrix (csr_matrix) containaing the distance score between the names  
    linkage: string defining the linkage criteria to apply for the hierarchical clustering  
    
    Returns:
    linkage_matrix: numpy linkage matrix 
    """ 
    
    # Perform clustering
    clustering = AgglomerativeClustering(linkage=linkage, affinity='precomputed', compute_distances=True,
                                         distance_threshold=0.75, n_clusters=None)
    clustering.fit(distance_matrix)

    # Counts the leafs for each node
    counts = np.zeros(clustering.children_.shape[0])
    n_leafs = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_leafs:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_leafs]
        counts[i] = current_count

    # Create the linkage matrix
    linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)
    return linkage_matrix


def get_leafs_recursively(node_id, linkage_matrix, data):
    """
    Extract list of leaf elements from a given node id 
    
    Parameters:
    node_id: (int) node id 
    linkage_matrix: numpy linkage matrix from the clustering
    data: pandas serie with cleaned names
    
    Returns:
    leafs: (list of string) list of the names 
    """ 
    
    leafs = []
    n_names = data.shape[0]
    
    # Single-element cluster
    if node_id < n_names:
        leafs.append(data.at[node_id])
        
    # Call recursively the function for the associated elements of the link
    else:
        leafs += get_leafs_recursively(int(linkage_matrix[node_id-n_names,0]), linkage_matrix, data)
        leafs += get_leafs_recursively(int(linkage_matrix[node_id-n_names,1]), linkage_matrix, data)

    return leafs


def get_common_sequences(leafs):
    """
    Find common sequences from a list of elements.
    Used to find the possible common entity names.
    
    Parameters:
    leafs: list of strings with element names 
    
    Returns:
    string: common sequences separated by semicolons 
    """ 
    
    leafs_length = len(leafs)
    
    # Clusters with more than one element 
    if leafs_length > 1 :
        commong_sequences = []
        
        # Loop over all element pairs
        for i in range(leafs_length):
            for j in range(i+1,leafs_length):
                string1, string2 = leafs[i], leafs[j]
                # Find the longuest common sequence
                match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
                if match.size != 0:
                    commong_sequences.append(string1[match.a: match.a + match.size].strip())
                    
        # Return the list of the three most-occuring sequences between the pairs
        counts = Counter(commong_sequences)
        return ";".join([i[0] for i in counts.most_common(3)])
    
    # Return the element name for single-element clusters
    else: 
        return leafs[0]

    
def get_dict_cluster_name(fclusters, linkage_matrix, data):
    """
    Define dictionary of possible entity names for each cluster.
    
    Parameters:
    fclusters: array of cluster id for each elements
    linkage_matrix: numpy linkage matrix from the clustering
    data: pandas serie with cleaned names
    
    Returns:
    dict_cluster_name: dictionary mapping each cluster id to its associated possible entity names 
    """ 
        
    dict_cluster_name = {}
    
    # Get leader nodes for each clusters
    leaders_nodes, leaders_labels = leaders(linkage_matrix, fclusters)
    
    # Loop over leader nodes
    for node_id, cluster_id in zip(leaders_nodes, leaders_labels):
        # Extract leaf elements
        leafs = get_leafs_recursively(node_id, linkage_matrix, data)
        # Get possible entity names
        common_sequences = get_common_sequences(leafs)
        
        dict_cluster_name[cluster_id] = common_sequences
        
    return dict_cluster_name
    

def name_mapping(dfs, colname):
    """
    Run the clustering and mapping algorithm on the dataset.
    
    Parameters:
    data: pandas object containing the full dataset to clusterize
    colname: column name containing the company names
    
    Returns:
    data_output: pandas object re-arranged into clusters 
    """ 
    
    # Extract the relevant column
    data = dfs[colname]

    # Create output dataset
    dfs_output = dfs.copy()

    # Data pre-processing
    print("Pre-processing the data...")
    data_prep = data_preprocessing(data)

    # Perform clustering 
    print("Clustering...")
    distance_matrix = get_distance_matrix(data_prep)
    linkage_matrix = get_linkage_matrix(distance_matrix)

    # Loop over cluster layers
    threshold_list = [0.35, 0.5, 0.65]
    for thld in threshold_list:
        print("Mapping using a threshold distance of "+str(thld)+" ...")
        fclusters = fcluster(linkage_matrix, thld, criterion='distance')
        dict_cluster_name = get_dict_cluster_name(fclusters, linkage_matrix, data_prep)
        dfs_output["Cluster ID "+str(thld)] =  dfs_output.apply(lambda row: fclusters[row.name], axis=1)
        dfs_output["Mapped name "+str(thld)] =  dfs_output.apply(lambda row: dict_cluster_name[fclusters[row.name]], axis=1)

    # Reorder the names following the clustering result
    print("Re-ordering the data...")
    names_ordered = leaves_list(linkage_matrix)
    dfs_output = dfs_output.reindex(names_ordered)

    return dfs_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Map the company names to their common entity.')
    parser.add_argument('infile', help="Excel input file containing the company names.")
    parser.add_argument('colname', help="Column name containing the company names.")
    parser.add_argument("-o", "--output", dest='oufile', default='Result.xlsx', help='Output file name (Default: Result.xlsx).')

    args = parser.parse_args()
    
    print("Reading intput file...")
    dfs = pd.read_excel(args.infile)

    dfs_output = name_mapping(dfs, args.colname)

    print("Mapping result saved in : "+args.oufile)
    dfs_output.to_excel(args.oufile)
