import os
import pdb
import torch
import numpy as np
import pandas as pd

class UniqueGeneNetAnalyse():
    def __init__(self):
        pass

    def intersect_net(self, node_threshold):
        # READ INFORMATION FROM FILE
        gene_up_weight_edge_df = pd.read_csv('./analysis_nci/averaged_fold_kegg_weighted_gene_interaction.csv')
        gene_up_degree_df = pd.read_csv('./analysis_nci/bianalyse_data/gene_weight_up_degree.csv')
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('./analysis_nci/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}

        # BUILD UP NODES DEGREE [DELETION LIST]
        filter_node_count = 0
        up_node_deletion_index_list = []
        up_node_index_list = []
        up_node_name_list = []
        up_node_degree_list = []
        for up_row in gene_up_degree_df.itertuples():
            if list(up_row[5:])[0] <= node_threshold or up_row[1] == 1572:
                up_node_deletion_index_list.append(up_row[1])
            else:
                filter_node_count += 1
                up_node_name_list.append(up_row[2])
                up_node_index_list.append(up_row[1])
                up_node_degree_list.append(up_row[5])
        print('----- FILTERED UP GENES WITH HIGHER DEGREES: ' + str(filter_node_count) + ' -----')
        # REMOVE CERTAIN DELETED GENES IN [cellline_gene_num_dict]
        up_filter_gene_dict = cellline_gene_num_dict
        [up_filter_gene_dict.pop(key) for key in up_node_deletion_index_list]

        # READ INFORMATION FROM FILE
        dir_opt = self.dir_opt
        gene_down_weight_edge_df = pd.read_csv('./analysis_nci/bianalyse_data/gene_down_weight_edge.csv')
        gene_down_degree_df = pd.read_csv('./analysis_nci/bianalyse_data/gene_weight_down_degree.csv')
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('./analysis_nci/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}

        # BUILD DOWN NODES DEGREE [DELETION LIST]
        filter_node_count = 0
        down_node_deletion_index_list = []
        down_node_index_list = []
        down_node_name_list = []
        down_node_degree_list = []
        for down_row in gene_down_degree_df.itertuples():
            if list(down_row[5:])[0] <= node_threshold :
                down_node_deletion_index_list.append(down_row[1])
            else:
                filter_node_count += 1
                down_node_name_list.append(down_row[2])
                down_node_index_list.append(down_row[1])
                down_node_degree_list.append(down_row[5])
        print('----- FILTERED DOWN GENES WITH HIGHER DEGREES: ' + str(filter_node_count) + ' -----')
        # REMOVE CERTAIN DELETED GENES IN [cellline_gene_num_dict]
        down_filter_gene_dict = cellline_gene_num_dict
        [down_filter_gene_dict.pop(key) for key in down_node_deletion_index_list]

        
        up_node_name_set = set(up_node_name_list)
        down_node_name_set = set(down_node_name_list)
        up_down_intersect = up_node_name_set.intersection(down_node_name_set)
        up_down_intersect_list = list(up_down_intersect)
        print('----- UPSTREAM AND DOWNSTREAM INTERSECTED GENES: -----')
        print(len(up_down_intersect_list))
        # up_down_intersect_list = sorted(up_down_intersect_list)
        print(up_down_intersect_list)
        up_down_intersect_df = pd.DataFrame(up_down_intersect_list, columns = ['gene_name'])
        up_down_intersect_df.to_csv('./analysis_nci/post_data/up_down_intersect.csv', index = False, header = True)
        # UPSTREAM UNIQUE GENES
        up_node_unique_list = []
        for node in up_node_name_list:
            if node not in up_down_intersect_list:
                up_node_unique_list.append(node)
        print('----- UPSTREAM UNIQUE GENES: -----')
        print(len(up_node_unique_list))
        up_unique_df = pd.DataFrame(up_node_unique_list, columns = ['gene_name'])
        up_unique_df.to_csv('./analysis_nci/post_data/up_unique.csv', index = False, header = True)
        # up_node_unique_list = sorted(up_node_unique_list)
        print(up_node_unique_list)
        # DOWNSTREAM UNIQUE GENES
        down_node_unique_list = []
        for node in down_node_name_list:
            if node not in up_down_intersect_list:
                down_node_unique_list.append(node)
        print('----- DOWNSTREAM UNIQUE GENES: -----')
        print(len(down_node_unique_list))
        # down_node_unique_list = sorted(down_node_unique_list)
        print(down_node_unique_list)
        down_unique_df = pd.DataFrame(down_node_unique_list, columns = ['gene_name'])
        down_unique_df.to_csv('./analysis_nci/post_data/down_unique.csv', index = False, header = True)



if __name__ == "__main__":
    # BASICAL PARAMETERS IN FILES
    dir_opt = '/datainfo2'
    node_threshold = 30.0
    # ANALYSE THE UNIQUE GENES IN [Upstream/Downstream] 
    UniqueGeneNetAnalyse(dir_opt).intersect_net(node_threshold)

