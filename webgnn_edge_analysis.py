import os
import pdb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from geo_tmain_webgnn import arg_parse, build_geowebgnn_model

class PanEdgeAnalyse():
    def __init__(self):
        pass

    def reform_weight_adj(self, fold_n, model):
        print('\nLOADING WEIGHT PARAMETERS FROM SAVED MODEL...')
        # COLLECT WEIGHT
        # import pdb; pdb.set_trace()

        first_conv_up_weight = model.conv_first.up_gene_edge_weight.cpu().data.numpy()
        first_conv_down_weight = model.conv_first.down_gene_edge_weight.cpu().data.numpy()
        block_conv_up_weight = model.conv_block.up_gene_edge_weight.cpu().data.numpy()
        block_conv_down_weight = model.conv_block.down_gene_edge_weight.cpu().data.numpy()
        last_conv_up_weight = model.conv_last.up_gene_edge_weight.cpu().data.numpy()
        last_conv_down_weight = model.conv_last.down_gene_edge_weight.cpu().data.numpy()
        if os.path.exists('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan') == False:
            os.mkdir('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan')
        np.save('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan/first_conv_up_weight.npy', first_conv_up_weight)
        np.save('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan/first_conv_down_weight.npy', first_conv_down_weight)
        np.save('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan/block_conv_up_weight.npy', block_conv_up_weight)
        np.save('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan/block_conv_down_weight.npy', block_conv_down_weight)
        np.save('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan/last_conv_up_weight.npy', last_conv_up_weight)
        np.save('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan/last_conv_down_weight.npy', last_conv_down_weight)

        # import pdb; pdb.set_trace()
        # MAKE ABSOLUTE VALUE
        first_conv_up_weight = np.absolute(first_conv_up_weight)
        first_conv_down_weight = np.absolute(first_conv_down_weight)
        block_conv_up_weight = np.absolute(block_conv_up_weight)
        block_conv_down_weight = np.absolute(block_conv_down_weight)
        last_conv_up_weight = np.absolute(last_conv_up_weight)
        last_conv_down_weight = np.absolute(last_conv_down_weight)
        conv_up_weight = (1/3) * (first_conv_up_weight + block_conv_up_weight + last_conv_up_weight)
        conv_down_weight = (1/3) * (first_conv_down_weight + block_conv_down_weight + last_conv_down_weight)
        conv_bind_weight = (1/2) * (conv_up_weight + conv_down_weight)

        # COMBINE WITH [kegg_gene_interaction.csv]
        kegg_gene_interaction_df = pd.read_csv('./data/filtered_data/kegg_gene_interaction.csv')
        kegg_gene_interaction_df['conv_bind_weight'] = conv_bind_weight

        # node_num_dict_df = pd.read_csv('./analysis_' + dataset + '/node_num_dict.csv')
        kegg_gene_num_dict_df = pd.read_csv('./data/filtered_data/kegg_gene_num_dict.csv')
        kegg_gene_num_dict_df = kegg_gene_num_dict_df[['gene_num', 'kegg_gene']]
        kegg_gene_num_dict_df = kegg_gene_num_dict_df.rename(columns={'gene_num': 'node_num', 'kegg_gene': 'node_name'})
        kegg_gene_num_dict_df['node_type'] = ['gene'] * (kegg_gene_num_dict_df.shape[0])
        drug_num_dict_df = pd.read_csv('./data/filtered_data/drug_num_dict.csv')
        drug_num_dict_df = drug_num_dict_df[['drug_num', 'Drug']]
        drug_num_dict_df = drug_num_dict_df.rename(columns={'drug_num': 'node_num', 'Drug': 'node_name'})
        drug_num_dict_df['node_type'] = ['drug'] * (drug_num_dict_df.shape[0])
        # import pdb; pdb.set_trace()
        node_num_dict_df = pd.concat([kegg_gene_num_dict_df, drug_num_dict_df])
        node_num_dict_df.to_csv('./analysis_' + dataset + '/node_num_dict.csv', header=True, index=False)
        node_num_dict = dict(zip(node_num_dict_df.node_name, node_num_dict_df.node_num))
        kegg_gene_interaction_df = kegg_gene_interaction_df.replace({'src': node_num_dict, 'dest': node_num_dict})
        kegg_gene_interaction_df.to_csv('./analysis_' + dataset + '/fold_' + str(fold_n) + '_pan/kegg_weighted_gene_interaction.csv', index=False, header=True)


    def average_nfold(self):
        fold_1_weight_df = pd.read_csv('./analysis_nci/fold_1_pan/kegg_weighted_gene_interaction.csv')
        fold_2_weight_df = pd.read_csv('./analysis_nci/fold_2_pan/kegg_weighted_gene_interaction.csv')
        fold_3_weight_df = pd.read_csv('./analysis_nci/fold_3_pan/kegg_weighted_gene_interaction.csv')
        fold_4_weight_df = pd.read_csv('./analysis_nci/fold_4_pan/kegg_weighted_gene_interaction.csv')
        fold_5_weight_df = pd.read_csv('./analysis_nci/fold_5_pan/kegg_weighted_gene_interaction.csv')

        import pdb; pdb.set_trace()
        averaged_fold_df = pd.concat([fold_1_weight_df, fold_2_weight_df, fold_3_weight_df, fold_4_weight_df, fold_5_weight_df]).groupby(level=0).mean()
        averaged_fold_df.to_csv('./analysis_nci/averaged_fold_kegg_weighted_gene_interaction.csv', index=False, header=True)


if __name__ == "__main__":
    # ##### REBUILD MODEL AND ANALYSIS PARAMTERS
    # prog_args = arg_parse()
    # device = torch.device('cuda:0') 
    # model = build_geowebgnn_model(prog_args, device)

    # # SET THE FOLD FOR MODEL
    # fold_n = 5
    # # dataset = 'oneil'
    # dataset = 'nci'
    # load_path = './data/result/' + dataset + '_webgnn/epoch_200_4/best_train_model.pt'
    # model.load_state_dict(torch.load(load_path, map_location=device))

    # if os.path.exists('./analysis_' + dataset) == False:
    #     os.mkdir('./analysis_' + dataset)
    # PanEdgeAnalyse().reform_weight_adj(fold_n, model)

    PanEdgeAnalyse().average_nfold()
