import os
import pdb
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 

from geo_tmain_webgnn import arg_parse, build_geowebgnn_model

class ReformWeightAdj():
    def __init__(self):
        pass

    def form_node_degree(self):
        # MAP GENES WITH INDEX NUM !!! START FROM 1
        kegg_gene_num_dict_df = pd.read_csv('./data/filtered_data/kegg_gene_num_dict.csv')
        kegg_gene_num_dict = dict(zip(kegg_gene_num_dict_df.kegg_gene, kegg_gene_num_dict_df.gene_num))
        # CALCULATE NODE DEGREE
        averaged_fold_df = pd.read_csv('./analysis_nci/averaged_fold_kegg_weighted_gene_interaction.csv')
        G = nx.Graph()
        for row in averaged_fold_df.itertuples():
            src_idx = row[1]
            dest_idx = row[2]
            bind_weight = row[3]
            G.add_edge(src_idx, dest_idx, weight=bind_weight)
        node_degree_dict = dict(sorted(G.degree(weight='weight')))
        node_weight_deg_df = pd.DataFrame({
            'gene_idx': node_degree_dict.keys(),
            'gene_name': list(kegg_gene_num_dict_df.kegg_gene),
            'degree': node_degree_dict.values()
        })
        print('\n-------- BINDSTREAM DEGREE --------')
        print(node_weight_deg_df)
        node_weight_deg_df.to_csv('./analysis_nci/gene_weight_bind_degree.csv', index=False, header=True)

    def filter_edge(self, edge_threshold):
        gene_bind_weight_edge_df = pd.read_csv('./analysis_nci/gene_bind_weight_edge.csv')
        bind_deletion_list = []
        # DELETE BIND EDGES SMALLER THAN CERTAIN THRESHOLD
        for bind_row in gene_bind_weight_edge_df.itertuples():
            if bind_row[3] < edge_threshold :
                bind_deletion_list.append(bind_row[0])
        edgefilter_gene_bind_weight_edge_df = gene_bind_weight_edge_df.drop(gene_bind_weight_edge_df.index[bind_deletion_list]).reset_index(drop=True)     
        edgefilter_gene_bind_weight_edge_df.to_csv('./analysis_nci/edgefilter_gene_bind_weight_edge.csv', index=False, header=True)

    def filter_gene(self, degree_threshold):
        # BIND DEGREE AND EDGE
        gene_weight_bind_degree_df = pd.read_csv('./analysis_nci/gene_weight_bind_degree.csv')
        gene_bind_weight_edge_df = pd.read_csv('./analysis_nci/gene_bind_weight_edge.csv')
        # FILTER SMALL DEGREE NODES
        node_bind_deletion_list = []
        edge_bind_deletion_list = []
        bind_num_deletion_list = []
        # DELETE BIND EDGES SMALLER THAN CERTAIN THRESHOLD
        for bind_row in gene_weight_bind_degree_df.itertuples():
            if bind_row[3] < degree_threshold :
                node_bind_deletion_list.append(bind_row[0])
                bind_num_deletion_list.append(bind_row[1])
        gene_weight_bind_degree_df = gene_weight_bind_degree_df.drop(gene_weight_bind_degree_df.index[node_bind_deletion_list]).reset_index(drop=True)     
        # print(gene_weight_bind_degree_df)
        for bind_row in gene_bind_weight_edge_df.itertuples(): 
            if bind_row[1] in bind_num_deletion_list or bind_row[2] in bind_num_deletion_list:
                edge_bind_deletion_list.append(bind_row[0])
        nodefilter_gene_bind_weight_edge_df = gene_bind_weight_edge_df.drop(gene_bind_weight_edge_df.index[edge_bind_deletion_list]).reset_index(drop=True)     
        nodefilter_gene_bind_weight_edge_df.to_csv('./analysis_nci/nodefilter_gene_bind_weight_edge.csv', index=False, header=True)



if __name__ == "__main__":
    # ###########################################################################################
    # ########################### REBUILD MODEL AND ANALYSIS PARAMTERS ##########################
    # ###########################################################################################
    # ReformWeightAdj().form_node_degree()


    # ###########################################################################################
    # ################################# FILTER EDGES OR NODES ###################################
    # ###########################################################################################
    edge_threshold = 0.1
    ReformWeightAdj().filter_edge(edge_threshold)
    degree_threshold = 2
    ReformWeightAdj().filter_gene(degree_threshold)


    