import os
import torch
import numpy as np
import pandas as pd

from numpy import inf
from scipy import sparse
from sklearn.model_selection import train_test_split

class LoadData():
    def __init__(self, dataset):
        self.dataset = dataset
        pass

    def load_batch(self, index, upper_index, place_num, drug_feature=False):
        # PRELOAD EACH SPLIT DATASET
        split_input_df = pd.read_csv('./' + self.dataset+ '/filtered_data/split_input_' + str(place_num + 1) + '.csv')
        num_feature = 4
        final_annotation_gene_df = pd.read_csv('./' + self.dataset+ '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        print('READING GENE FEATURES FILES ...')
        final_gdsc_rna_df = pd.read_csv('./' + self.dataset+ '/filtered_data/final_rna.csv')
        final_gdsc_cnv_df = pd.read_csv('./' + self.dataset+ '/filtered_data/final_cnv.csv')
        num_gene, num_cellline = final_gdsc_rna_df.shape
        # CONVERT [drugbank.csv] TO A LIST
        print('READING DRUGBANK ...')
        final_drugbank_df = pd.read_csv('./' + self.dataset+ '/filtered_data/final_drugbank.csv')
        final_drugbank_comlist = final_drugbank_df.values.tolist()
        print('READING DRUGDICT ...')
        drug_num_dict_df = pd.read_csv('./' + self.dataset+ '/filtered_data/drug_num_dict.csv')
        drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
        num_drug = len(drug_dict)
        print('READING FINISHED ...')
        # COMBINE A BATCH SIZE AS [x_batch, y_batch, drug_batch]
        print('-----' + str(index) + ' to ' + str(upper_index) + '-----')
        tmp_batch_size = 0
        y_input_list = []
        drug_input_list = []
        x_batch = np.zeros((1, (num_feature * (num_gene + num_drug))))
        for row in split_input_df.iloc[index : upper_index].itertuples():
            tmp_batch_size += 1
            drug_a = row[1]
            drug_b = row[2]
            cellline_name = row[3]
            if dataset == 'data-drugcomb-fi':
                y = row[5]
            else:
                y = row[4]
            # DRUG_A AND [4853] TARGET GENES
            one_drug_target_list = []
            duo_drug_target_list = []
            for gene in gene_name_list:
                drugA_target = [drug_a, gene] in final_drugbank_comlist
                drugB_target = [drug_b, gene] in final_drugbank_comlist
                if drugA_target and drugB_target:
                    one_drug_target_list.append(0.0)
                    duo_drug_target_list.append(1.0)
                elif drugA_target or drugB_target:
                    one_drug_target_list.append(1.0)
                    duo_drug_target_list.append(0.0)
                else:
                    one_drug_target_list.append(0.0)
                    duo_drug_target_list.append(0.0)
            # GENE FEATURES SEQUENCE
            gene_rna_list = [float(x) for x in list(final_gdsc_rna_df[cellline_name])]
            gene_cnv_list = [float(x) for x in list(final_gdsc_cnv_df[cellline_name])]
            # COMBINE [drugA, drugB, rna, cmeth] 
            x_input_list = []
            for i in range(num_gene):
                # APPEND DRUG INFORMATION
                x_input_list.append(one_drug_target_list[i])
                x_input_list.append(duo_drug_target_list[i])
                # APPEND GENE FEATURES
                x_input_list.append(gene_rna_list[i])
                x_input_list.append(gene_cnv_list[i])
            if drug_feature == False:
                fillin_list = [0.0] * num_feature
                for i in range(num_drug):
                    x_input_list += fillin_list
            x_input = np.array(x_input_list)
            x_batch = np.vstack((x_batch, x_input))
            # COMBINE DRUG[A/B] LIST
            drug_input_list.append(drug_dict[drug_a])
            drug_input_list.append(drug_dict[drug_b])
            # COMBINE SCORE LIST
            y_input_list.append(y)
        # import pdb; pdb.set_trace()
        x_batch = np.delete(x_batch, 0, axis = 0)
        y_batch = np.array(y_input_list).reshape(tmp_batch_size, 1)
        drug_batch = np.array(drug_input_list).reshape(tmp_batch_size, 2)
        print(x_batch.shape)
        print(y_batch.shape)
        print(drug_batch.shape)
        return x_batch, y_batch, drug_batch


    def load_all_split(self, batch_size, k):
        form_data_path = './' + self.dataset+ '/form_data'
        # LOAD 100 PERCENT DATA
        print('LOADING ALL SPLIT DATA...')
        # FIRST LOAD EACH SPLIT DATA
        for place_num in range(k):
            split_input_df = pd.read_csv('./' + self.dataset+ '/filtered_data/split_input_' + str(place_num + 1) + '.csv')
            input_num, input_dim = split_input_df.shape
            num_feature = 4
            final_annotation_gene_df = pd.read_csv('./' + self.dataset+ '/filtered_data/kegg_gene_annotation.csv')
            gene_name_list = list(final_annotation_gene_df['kegg_gene'])
            num_gene = len(gene_name_list)
            drug_num_dict_df = pd.read_csv('./' + self.dataset+ '/filtered_data/drug_num_dict.csv')
            drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
            num_drug = len(drug_dict)
            x_split = np.zeros((1, num_feature * (num_gene + num_drug)))
            y_split = np.zeros((1, 1))
            drug_split = np.zeros((1, 2))
            for index in range(0, input_num, batch_size):
                if (index + batch_size) < input_num:
                    upper_index = index + batch_size
                else:
                    upper_index = input_num
                    datset = self.dataset
                x_batch, y_batch, drug_batch = LoadData(dataset).load_batch(index, upper_index, place_num)
                x_split = np.vstack((x_split, x_batch))
                y_split = np.vstack((y_split, y_batch))
                drug_split = np.vstack((drug_split, drug_batch))
            x_split = np.delete(x_split, 0, axis = 0)
            y_split = np.delete(y_split, 0, axis = 0)
            drug_split = np.delete(drug_split, 0, axis = 0)
            print('-------SPLIT DATA SHAPE-------')
            print(x_split.shape)
            print(y_split.shape)
            print(drug_split.shape)
            np.save(form_data_path + '/x_split' + str(place_num + 1) + '.npy', x_split)
            np.save(form_data_path + '/y_split' + str(place_num + 1) + '.npy', y_split)
            np.save(form_data_path + '/drug_split' + str(place_num + 1) + '.npy', drug_split)
            

    def load_train_test(self, k, n_fold):
        form_data_path = './' + self.dataset+ '/form_data'
        num_feature = 4
        final_annotation_gene_df = pd.read_csv('./' + self.dataset+ '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        num_gene = len(gene_name_list)
        drug_num_dict_df = pd.read_csv('./' + self.dataset+ '/filtered_data/drug_num_dict.csv')
        drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
        num_drug = len(drug_dict)
        xTr = np.zeros((1, num_feature * (num_gene + num_drug)))
        yTr = np.zeros((1, 1))
        drugTr = np.zeros((1, 2))
        for i in range(1, k + 1):
            if i == n_fold:
                print('--- LOADING ' + str(i) + '-TH SPLIT TEST DATA ---')
                xTe = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                yTe = np.load(form_data_path + '/y_split' + str(i) + '.npy')
                drugTe = np.load(form_data_path + '/drug_split' + str(i) + '.npy')
            else:
                print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
                x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
                drug_split = np.load(form_data_path + '/drug_split' + str(i) + '.npy')
                print('--- COMBINING DATA ... ---')
                xTr = np.vstack((xTr, x_split))
                yTr = np.vstack((yTr, y_split))
                drugTr = np.vstack((drugTr, drug_split))
        print('--- TRAINING INPUT SHAPE ---')
        xTr = np.delete(xTr, 0, axis = 0)
        yTr = np.delete(yTr, 0, axis = 0)
        drugTr = np.delete(drugTr, 0, axis = 0)
        print(xTr.shape)
        print(yTr.shape)
        print(drugTr.shape)
        np.save(form_data_path + '/xTr' + str(n_fold) + '.npy', xTr)
        np.save(form_data_path + '/yTr' + str(n_fold) + '.npy', yTr)
        np.save(form_data_path + '/drugTr' + str(n_fold) + '.npy', drugTr)
        print('--- TEST INPUT SHAPE ---')
        print(xTe.shape)
        print(yTe.shape)
        print(drugTe.shape)
        np.save(form_data_path + '/xTe' + str(n_fold) + '.npy', xTe)
        np.save(form_data_path + '/yTe' + str(n_fold) + '.npy', yTe)
        np.save(form_data_path + '/drugTe' + str(n_fold) + '.npy', drugTe)

    def combine_whole_dataset(self, k):
        form_data_path = './' + self.dataset+ '/form_data'
        num_feature = 4
        final_annotation_gene_df = pd.read_csv('./' + self.dataset+ '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        num_gene = len(gene_name_list)
        drug_num_dict_df = pd.read_csv('./' + self.dataset+ '/filtered_data/drug_num_dict.csv')
        drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
        num_drug = len(drug_dict)
        xAll = np.zeros((1, num_feature * (num_gene + num_drug)))
        yAll = np.zeros((1, 1))
        drugAll = np.zeros((1, 2))
        for i in range(1, k + 1):
            print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
            x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
            y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
            drug_split = np.load(form_data_path + '/drug_split' + str(i) + '.npy')
            print('--- COMBINING DATA ... ---')
            xAll = np.vstack((xAll, x_split))
            yAll = np.vstack((yAll, y_split))
            drugAll = np.vstack((drugAll, drug_split))
        print('--- ALL DATASET INPUT SHAPE ---')
        xAll = np.delete(xAll, 0, axis = 0)
        yAll = np.delete(yAll, 0, axis = 0)
        drugAll = np.delete(drugAll, 0, axis = 0)
        print(xAll.shape)
        print(yAll.shape)
        print(drugAll.shape)
        np.save(form_data_path + '/xAll.npy', xAll)
        np.save(form_data_path + '/yAll.npy', yAll)
        np.save(form_data_path + '/drugAll.npy', drugAll)
    
    def load_adj_edgeindex(self):
        form_data_path = './' + self.dataset+ '/form_data'
        # FORM A WHOLE ADJACENT MATRIX
        gene_num_df = pd.read_csv('./' + self.dataset+ '/filtered_data/kegg_gene_num_interaction.csv')
        src_gene_list = list(gene_num_df['src'])
        dest_gene_list = list(gene_num_df['dest'])
        final_annotation_gene_df = pd.read_csv('./' + self.dataset+ '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        num_gene = len(gene_name_list)
        dict_drug_num = pd.read_csv('./' + self.dataset+ '/filtered_data/drug_num_dict.csv')
        num_drug = dict_drug_num.shape[0]
        num_node = num_gene + num_drug
        adj = np.zeros((num_node, num_node))
        # GENE-GENE ADJACENT MATRIX
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i] - 1
            col_idx = dest_gene_list[i] - 1
            adj[row_idx, col_idx] = 1
            adj[col_idx, row_idx] = 1 # WHETHER WE WANT ['sym']
        # import pdb; pdb.set_trace()
        # DRUG_TARGET ADJACENT MATRIX
        drugbank_num_df = pd.read_csv('./' + self.dataset+ '/filtered_data/final_drugbank_num.csv')
        drugbank_drug_list = list(drugbank_num_df['Drug'])
        drugbank_target_list = list(drugbank_num_df['Target'])
        for row in drugbank_num_df.itertuples():
            row_idx = row[1] - 1
            col_idx = row[2] - 1
            adj[row_idx, col_idx] = 1
            adj[col_idx, row_idx] = 1
        # import pdb; pdb.set_trace()
        # np.save(form_data_path + '/adj.npy', adj)
        adj_sparse = sparse.csr_matrix(adj)
        sparse.save_npz(form_data_path + '/adj_sparse.npz', adj_sparse)
        # [edge_index]
        genedrug_src_list = src_gene_list + drugbank_drug_list + drugbank_target_list
        genedrug_dest_list = dest_gene_list + drugbank_target_list + drugbank_drug_list
        genedrug_src_indexlist = []
        genedrug_dest_indexlist = []
        for i in range(len(genedrug_src_list)):
            genedrug_src_index = genedrug_src_list[i] - 1
            genedrug_src_indexlist.append(genedrug_src_index)
            genedrug_dest_index = genedrug_dest_list[i] - 1
            genedrug_dest_indexlist.append(genedrug_dest_index)
        edge_index = np.column_stack((genedrug_src_indexlist, genedrug_dest_indexlist)).T
        np.save(form_data_path + '/edge_index.npy', edge_index)




# RANDOMIZE THE [final_NCI60_DeepLearningInput]
def input_random(dataset):
    final_input_df = pd.read_csv('./' + dataset + '/filtered_data/final_dl_input.csv')
    random_final_input_df = final_input_df.sample(frac = 1)
    random_final_input_df.to_csv('./' + dataset + '/filtered_data/random_final_dl_input.csv', index = False, header = True)

# SPLIT DEEP LEARNING INPUT INTO TRAINING AND TEST
def split_k_fold(k, dataset):
    random_final_dl_input_df = pd.read_csv('./' + dataset + '/filtered_data/random_final_dl_input.csv')
    num_points = random_final_dl_input_df.shape[0]
    num_div = int(num_points / k)
    num_div_list = [i * num_div for i in range(0, k)]
    num_div_list.append(num_points)
    # SPLIT [RandomFinal_NCI60_DeepLearningInput] INTO [k] FOLDS
    for place_num in range(k):
        low_idx = num_div_list[place_num]
        high_idx = num_div_list[place_num + 1]
        print('\n--------TRAIN-TEST SPLIT WITH TEST FROM ' + str(low_idx) + ' TO ' + str(high_idx) + '--------')
        split_input_df = random_final_dl_input_df[low_idx : high_idx]
        split_input_df.to_csv('./' + dataset + '/filtered_data/split_input_' + str(place_num + 1) + '.csv', index = False, header = True)



if __name__ == '__main__':
    # ############## MOUDLE 1 ################
    k = 2
    # DATASET SELECTION
    # dataset = 'data-drugcomb-fi'
    # dataset = 'data-DrugCombDB'
    # dataset = 'data-nci'
    dataset = 'data-oneil'

    # input_random(dataset)
    # split_k_fold(k, dataset)

    if os.path.exists('./' + dataset + '/form_data') == False:
        os.mkdir('./' + dataset + '/form_data')
    batch_size = 64
    LoadData(dataset).load_all_split(batch_size, k)

    n_fold = 2
    # ############## MOUDLE 2 ################
    print('split_input_' + str(n_fold) + '.csv')
    LoadData(dataset).load_adj_edgeindex()

    ################ MOUDLE 3 ################
    # FORM N-TH FOLD TRAINING DATASET
    LoadData(dataset).load_train_test(k, n_fold)