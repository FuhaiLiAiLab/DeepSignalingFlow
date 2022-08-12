import os
import pdb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 

from tmain_webgnn_decoder import arg_parse, build_webgnn_model

class PlotMSECorr():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def rebuild_loss_pearson(self, path, epoch_num):
        epoch_loss_list = []
        epoch_pearson_list = []
        min_train_loss = 100
        min_train_id = 0
        for i in range(1, epoch_num + 1):
            train_df = pd.read_csv(path + '/TrainingPred_' + str(i) + '.txt', delimiter=',')
            score_list = list(train_df['Score'])
            pred_list = list(train_df['Pred Score'])
            epoch_loss = mean_squared_error(score_list, pred_list)
            epoch_loss_list.append(epoch_loss)
            epoch_pearson = train_df.corr(method = 'pearson')
            epoch_pearson_list.append(epoch_pearson['Pred Score'][0])
            if epoch_loss < min_train_loss:
                min_train_loss = epoch_loss
                min_train_id = i
        print('-------------BEST MODEL ID:' + str(min_train_id) + '-------------')
        print('BEST MODEL TRAIN LOSS: ', min_train_loss)
        print('BEST MODEL PEARSON CORR: ', epoch_pearson_list[min_train_id - 1])
        # print('\n-------------EPOCH TRAINING PEARSON CORRELATION LIST: -------------')
        # print(epoch_pearson_list)
        # print('\n-------------EPOCH TRAINING MSE LOSS LIST: -------------')
        # print(epoch_loss_list)
        epoch_pearson_array = np.array(epoch_pearson_list)
        epoch_loss_array = np.array(epoch_loss_list)
        np.save(path + '/pearson.npy', epoch_pearson_array)
        np.save(path + '/loss.npy', epoch_loss_array)
        return min_train_id

    def plot_loss_pearson(self, path, epoch_num):
        epoch_pearson_array = np.load(path + '/pearson.npy')
        epoch_loss_array = np.load(path + '/loss.npy')
        x = range(1, epoch_num + 1)
        plt.figure(1)
        plt.title('Training Loss and Pearson Correlation in ' + str(epoch_num) + ' Epochs') 
        plt.xlabel('Train Epochs') 
        plt.figure(1)
        plt.subplot(211)
        plt.plot(x, epoch_loss_array) 
        plt.subplot(212)
        plt.plot(x, epoch_pearson_array)
        plt.show()

    def plot_train_real_pred(self, path, best_model_num, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/TrainingPred_' + best_model_num + '.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Training Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TRAINING PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_train'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        if os.path.exists('.' + dir_opt + '/plot') == False:
            os.mkdir('.' + dir_opt + '/plot')
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)
        
    def plot_test_real_pred(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/TestPred.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Test Dataset'
        ax = pred_dl_input_df.plot(x = 'Score', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('Score')
        ax.set_ylabel('Pred Score')
        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_test'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)

class ReformWeightAdj():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    # FORM CERTAIN [weight_adj] IN CERTAIN [dataset_num]
    def reform_weight_adj(self, RNA_seq_filename, model, dataset_num, conv_concat):
        dir_opt = self.dir_opt
        # WEIGHT PARAMETETS IN MODEL, ALSO (Src -> Dest)
        if model:
            print('\nLOADING WEIGHT PARAMETERS FROM SAVED MODEL...')
            first_conv_up_weight = model.conv_first.up_weight_adj.cpu().data.numpy()
            first_conv_down_weight = model.conv_first.down_weight_adj.cpu().data.numpy()
            block_conv_up_weight = model.conv_block[0].up_weight_adj.cpu().data.numpy()
            block_conv_down_weight = model.conv_block[0].down_weight_adj.cpu().data.numpy()
            last_conv_up_weight = model.conv_last.up_weight_adj.cpu().data.numpy()
            last_conv_down_weight = model.conv_last.down_weight_adj.cpu().data.numpy()
            if os.path.exists('.' + dir_opt + '/bianalyse_data') == False:
                os.mkdir('.' + dir_opt + '/bianalyse_data')
            if os.path.exists('.' + dir_opt + '/bianalyse_data/' + dataset_num) == False:
                os.mkdir('.' + dir_opt + '/bianalyse_data/' + dataset_num)
            np.save('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/first_conv_up_weight.npy', first_conv_up_weight)
            np.save('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/first_conv_down_weight.npy', first_conv_down_weight)
            np.save('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/block_conv_up_weight.npy', block_conv_up_weight)
            np.save('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/block_conv_down_weight.npy', block_conv_down_weight)
            np.save('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/last_conv_up_weight.npy', last_conv_up_weight)
            np.save('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/last_conv_down_weight.npy', last_conv_down_weight)
            # print(first_conv_up_weight)
            # print(block_conv_up_weight)
            # print(last_conv_up_weight)
        else:
            print('\nLOADING WEIGHT FROM SAVE NUMPY FILES...')
            first_conv_up_weight = np.load('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/first_conv_up_weight.npy')
            first_conv_down_weight = np.load('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/first_conv_down_weight.npy')
            block_conv_up_weight = np.load('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/block_conv_up_weight.npy')
            block_conv_down_weight = np.load('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/block_conv_down_weight.npy')
            last_conv_up_weight = np.load('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/last_conv_up_weight.npy')
            last_conv_down_weight = np.load('.' + dir_opt + '/bianalyse_data/' + dataset_num + '/last_conv_down_weight.npy')

        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}
        # GENE-GENE ADJACENT MATRIX (Src -> Dest)
        form_data_path = '.' + dir_opt + '/form_data'
        cellline_gene_num_df = pd.read_csv(form_data_path + '/gene_connection_num.txt')
        src_gene_list = list(cellline_gene_num_df['src'])
        dest_gene_list = list(cellline_gene_num_df['dest'])
        num_gene = len(cellline_gene_df)
        num_node = num_gene + 2
        gene_adj = np.zeros((num_node, num_node))
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i] - 1
            col_idx = dest_gene_list[i] - 1
            gene_adj[row_idx, col_idx] = 1
        
        # WEIGHTING [Absolute Value] ADJACENT GENE-GENE MATRICES (Src -> Dest)
        first_conv_up_weight_adj = np.multiply(gene_adj, np.absolute(first_conv_up_weight))
        block_conv_up_weight_adj = np.multiply(gene_adj, np.absolute(block_conv_up_weight))
        last_conv_up_weight_adj = np.multiply(gene_adj, np.absolute(last_conv_up_weight))
        # WEIGHTING [Absolute Value] ADJACENT GENE-GENE MATRICES (Dest -> Src)
        first_conv_down_weight_adj = np.multiply(gene_adj, np.absolute(first_conv_down_weight))
        block_conv_down_weight_adj = np.multiply(gene_adj, np.absolute(block_conv_down_weight))
        last_conv_down_weight_adj = np.multiply(gene_adj, np.absolute(last_conv_down_weight))
        # OPTION ON CONCATING WEIGHT ADJ IN [first, block, last] LAYERS
        if conv_concat == False:
            conv_up_weight_adj = first_conv_up_weight_adj
            conv_down_weight_adj = first_conv_down_weight_adj
        else:
            conv_up_weight_adj = (1/3) * (first_conv_up_weight_adj + block_conv_up_weight_adj + last_conv_up_weight_adj)
            conv_down_weight_adj = (1/3) * (first_conv_down_weight_adj + block_conv_down_weight_adj + last_conv_down_weight_adj)
        return conv_up_weight_adj, conv_down_weight_adj
    
    
    def form_weight_edge(self, RNA_seq_filename, dataset_num_list, conv_concat):
        dir_opt = self.dir_opt
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}
        print(cellline_gene_num_dict)
        # GENE-GENE ADJACENT MATRIX (Src -> Dest)
        form_data_path = '.' + dir_opt + '/form_data'
        cellline_gene_num_df = pd.read_csv(form_data_path + '/gene_connection_num.txt')
        src_gene_list = list(cellline_gene_num_df['src'])
        dest_gene_list = list(cellline_gene_num_df['dest'])
        num_gene = len(cellline_gene_df)
        num_node = num_gene + 2
        gene_adj = np.zeros((num_node, num_node))
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i] - 1
            col_idx = dest_gene_list[i] - 1
            gene_adj[row_idx, col_idx] = 1
        # COMBINE ALL DATASET [conv_up, conv_down]
        count = 0
        conv_up_weight_adj_bind = np.zeros((num_node, num_node))
        conv_down_weight_adj_bind = np.zeros((num_node, num_node))
        for dataset_num in dataset_num_list:
            if count == 0:
                load_path = './datainfo2/result/epoch_75/best_train_model.pth'
            else:
                load_path = './datainfo2/result/epoch_75_' + str(count) + '/best_train_model.pth'
            count += 1
            prog_args = arg_parse()
            model = build_webgnn_model(prog_args)
            model.load_state_dict(torch.load(load_path))
            conv_up_weight_adj, conv_down_weight_adj = ReformWeightAdj(dir_opt).reform_weight_adj(RNA_seq_filename, model, dataset_num, conv_concat)
            conv_up_weight_adj_bind += conv_up_weight_adj
            conv_down_weight_adj_bind += conv_down_weight_adj
        conv_up_weight_adj_bind = (1 / count) * conv_up_weight_adj_bind
        conv_down_weight_adj_bind = (1 / count) * conv_down_weight_adj_bind

        # CONVERT ADJACENT MATRICES TO EDGES, GENE ID STARTS WITH 1 (UP WITH src -> dest) (DOWN WITH dest -> src)
        up_src_gene_name_list = []
        up_dest_gene_name_list = []
        up_src_gene_num_list = []
        up_dest_gene_num_list = []
        up_weight_edge_list = []
        down_src_gene_name_list = []
        down_dest_gene_name_list = []
        down_src_gene_num_list = []
        down_dest_gene_num_list = []
        down_weight_edge_list = []
        for row in range(num_gene):
            for col in range(num_gene):
                if gene_adj[row, col]:
                    # RECORD FOR [src -> dest]
                    up_src_gene_name_list.append(cellline_gene_num_dict[row + 1])
                    up_dest_gene_name_list.append(cellline_gene_num_dict[col + 1])
                    up_src_gene_num_list.append(row + 1)
                    up_dest_gene_num_list.append(col + 1)
                    # RECORD FOR [dest -> src]
                    down_src_gene_name_list.append(cellline_gene_num_dict[col + 1])
                    down_dest_gene_name_list.append(cellline_gene_num_dict[row + 1])
                    down_src_gene_num_list.append(col + 1)
                    down_dest_gene_num_list.append(row + 1)
                    # ########ONLY USE FIRST CONV WEIGHT########
                    up_weight_edge = conv_up_weight_adj_bind[row, col]
                    up_weight_edge_list.append(up_weight_edge)
                    down_weight_edge = conv_down_weight_adj_bind[row, col]
                    down_weight_edge_list.append(down_weight_edge)
        up_weight_src_dest = {'src': up_src_gene_num_list, 'src_name': up_src_gene_name_list, \
                        'dest': up_dest_gene_num_list, 'dest_name': up_dest_gene_name_list, \
                        'weight': up_weight_edge_list}
        down_weight_src_dest = {'src': down_src_gene_num_list, 'src_name': down_src_gene_name_list, \
                        'dest': down_dest_gene_num_list, 'dest_name': down_dest_gene_name_list, \
                        'weight': down_weight_edge_list}
        gene_up_weight_edge_df = pd.DataFrame(up_weight_src_dest)
        gene_down_weight_edge_df = pd.DataFrame(down_weight_src_dest)
        print('\n--------UPSTREAM TO DOWNSTREAM GENE EDGES WEIGHT--------')
        print(gene_up_weight_edge_df)
        print('\n--------DOWNSTREAM TO UPSTREAM GENE EDGES WEIGHT--------')
        print(gene_down_weight_edge_df)
        gene_up_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_up_weight_edge.csv', index = False, header = True)
        gene_down_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_down_weight_edge.csv', index = False, header = True)

        # CONVERT ADJACENT MATRICES TO EDGES, GENE ID STARTS WITH 1 (BIND WITH UP/DOWN)
        conv_mean_weight_adj_bind = (1/2) * (conv_up_weight_adj_bind + conv_down_weight_adj_bind)
        bind_src_gene_name_list = []
        bind_dest_gene_name_list = []
        bind_src_gene_num_list = []
        bind_dest_gene_num_list = []
        bind_weight_edge_list = []
        for row in range(num_gene):
            for col in range(num_gene):
                if gene_adj[row, col]:
                    # RECORD FOR [src -> dest]
                    bind_src_gene_name_list.append(cellline_gene_num_dict[row + 1])
                    bind_dest_gene_name_list.append(cellline_gene_num_dict[col + 1])
                    bind_src_gene_num_list.append(row + 1)
                    bind_dest_gene_num_list.append(col + 1)
                    mean_weight_edge = conv_mean_weight_adj_bind[row, col]
                    bind_weight_edge_list.append(mean_weight_edge)
        bind_weight_src_dest = {'src': bind_src_gene_num_list, 'src_name': bind_src_gene_name_list, \
                        'dest': bind_dest_gene_num_list, 'dest_name': bind_dest_gene_name_list, \
                        'weight': bind_weight_edge_list}
        gene_bind_weight_edge_df = pd.DataFrame(bind_weight_src_dest)
        print('\n--------BIND-STREAM GENE EDGES WEIGHT--------')
        print(gene_bind_weight_edge_df)
        gene_bind_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_bind_weight_edge.csv', index = False, header = True)
        return conv_up_weight_adj_bind, conv_down_weight_adj_bind, conv_mean_weight_adj_bind

    def form_node_degree(self, conv_up_weight_adj_bind, conv_down_weight_adj_bind, conv_mean_weight_adj_bind):
        dir_opt = self.dir_opt
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}
        num_gene = len(cellline_gene_df)
        num_node = num_gene + 2
        # CONVERT ADJACENT MATRICES WITH GENE OUT/IN DEGREE
        gene_list = []
        gene_name_list = []
        gene_up_outdeg_list = []
        gene_up_indeg_list = []
        gene_up_degree_list = []
        gene_down_outdeg_list = []
        gene_down_indeg_list = []
        gene_down_degree_list = []
        gene_bind_outdeg_list = []
        gene_bind_indeg_list = []
        gene_bind_degree_list = []
        for idx in range(num_gene):
            gene_list.append(idx + 1)
            gene_name_list.append(cellline_gene_num_dict[idx + 1])
            # ######## OUT DEGREE ########
            gene_up_outdeg = np.sum(conv_up_weight_adj_bind[idx, :])
            gene_up_outdeg_list.append(gene_up_outdeg)
            gene_down_outdeg = np.sum(conv_down_weight_adj_bind[idx, :])
            gene_down_outdeg_list.append(gene_down_outdeg)
            gene_bind_outdeg = np.sum(conv_mean_weight_adj_bind[idx, :])
            gene_bind_outdeg_list.append(gene_bind_outdeg)
            # ######## IN DEGREE ########
            gene_up_indeg = np.sum(conv_up_weight_adj_bind[:, idx])
            gene_up_indeg_list.append(gene_up_indeg)
            gene_down_indeg = np.sum(conv_down_weight_adj_bind[:, idx])
            gene_down_indeg_list.append(gene_down_indeg)
            gene_bind_indeg = np.sum(conv_mean_weight_adj_bind[:, idx])
            gene_bind_indeg_list.append(gene_bind_indeg)
            # ######## DEGREE ########
            gene_up_degree = gene_up_outdeg + gene_up_indeg
            gene_down_degree = gene_down_outdeg + gene_down_indeg
            gene_bind_degree = gene_bind_outdeg + gene_bind_indeg
            gene_up_degree_list.append(gene_up_degree)
            gene_down_degree_list.append(gene_down_degree)
            gene_bind_degree_list.append(gene_bind_degree)
        # ######## UP DATAFRAME ########
        weight_up_gene_degree = {'gene_idx': gene_list, 'gene_name': gene_name_list, 'out_degree': gene_up_outdeg_list,\
                        'in_degree': gene_up_indeg_list, 'degree': gene_up_degree_list}
        gene_weight_up_degree_df = pd.DataFrame(weight_up_gene_degree)
        print('\n-------- UPSTREAM DEGREE --------')
        print(gene_weight_up_degree_df)
        gene_weight_up_degree_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_weight_up_degree.csv', index = False, header = True)
        # ######## DOWN DATAFRAME ########
        weight_down_gene_degree = {'gene_idx': gene_list, 'gene_name': gene_name_list, 'out_degree': gene_down_outdeg_list,\
                        'in_degree': gene_down_indeg_list, 'degree': gene_down_degree_list}
        gene_weight_down_degree_df = pd.DataFrame(weight_down_gene_degree)
        print('\n-------- DOWNSTREAM DEGREE --------')
        print(gene_weight_down_degree_df)
        gene_weight_down_degree_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_weight_down_degree.csv', index = False, header = True)
        # ######## BIND DATAFRAME ########
        weight_bind_gene_degree = {'gene_idx': gene_list, 'gene_name': gene_name_list, 'out_degree': gene_bind_outdeg_list,\
                        'in_degree': gene_bind_indeg_list, 'degree': gene_bind_degree_list}
        gene_weight_bind_degree_df = pd.DataFrame(weight_bind_gene_degree)
        print('\n-------- BINDSTREAM DEGREE --------')
        print(gene_weight_bind_degree_df)
        gene_weight_bind_degree_df.to_csv('.' + dir_opt + '/bianalyse_data/gene_weight_bind_degree.csv', index = False, header = True)

    def filter_edge(self, edge_threshold):
        dir_opt = self.dir_opt
        gene_up_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_up_weight_edge.csv')
        gene_down_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_down_weight_edge.csv')
        gene_bind_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_bind_weight_edge.csv')
        up_deletion_list = []
        down_deletion_list = []
        bind_deletion_list = []
        # DELETE UP EDGES SMALLER THAN CERTAIN THRESHOLD
        for up_row in gene_up_weight_edge_df.itertuples():
            if list(up_row[5:])[0] < edge_threshold :
                up_deletion_list.append(up_row[0])
        edgefilter_gene_up_weight_edge_df = gene_up_weight_edge_df.drop(gene_up_weight_edge_df.index[up_deletion_list]).reset_index(drop = True)     
        edgefilter_gene_up_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/edgefilter_gene_up_weight_edge.csv', index = False, header = True)
        # DELETE DOWN EDGES SMALLER THAN CERTAIN THRESHOLD
        for down_row in gene_up_weight_edge_df.itertuples():
            if list(down_row[5:])[0] < edge_threshold :
                down_deletion_list.append(down_row[0])
        edgefilter_gene_down_weight_edge_df = gene_down_weight_edge_df.drop(gene_down_weight_edge_df.index[down_deletion_list]).reset_index(drop = True)     
        edgefilter_gene_down_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/edgefilter_gene_down_weight_edge.csv', index = False, header = True)
        # DELETE BIND EDGES SMALLER THAN CERTAIN THRESHOLD
        for bind_row in gene_bind_weight_edge_df.itertuples():
            if list(bind_row[5:])[0] < edge_threshold :
                bind_deletion_list.append(bind_row[0])
        edgefilter_gene_bind_weight_edge_df = gene_bind_weight_edge_df.drop(gene_bind_weight_edge_df.index[bind_deletion_list]).reset_index(drop = True)     
        edgefilter_gene_bind_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/edgefilter_gene_bind_weight_edge.csv', index = False, header = True)

    def filter_gene(self, degree_threshold):
        dir_opt = self.dir_opt
        # UP DEGREE AND EDGE
        gene_weight_up_degree_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_weight_up_degree.csv')
        gene_up_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_up_weight_edge.csv')
        # DOWN DEGREE AND EDGE
        gene_weight_down_degree_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_weight_down_degree.csv')
        gene_down_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_down_weight_edge.csv')
        # BIND DEGREE AND EDGE
        gene_weight_bind_degree_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_weight_bind_degree.csv')
        gene_bind_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_bind_weight_edge.csv')
        # FILTER SMALL DEGREE NODES
        up_deletion_list = []
        up_num_deletion_list = []
        down_deletion_list = []
        down_num_deletion_list = []
        bind_deletion_list = []
        bind_num_deletion_list = []
        # DELETE UP EDGES SMALLER THAN CERTAIN THRESHOLD
        for up_row in gene_weight_up_degree_df.itertuples():
            if list(up_row[5:])[0] < degree_threshold :
                up_deletion_list.append(up_row[0])
                up_num_deletion_list.append(up_row[1:][0])
        gene_weight_up_degree_df = gene_weight_up_degree_df.drop(gene_weight_up_degree_df.index[up_deletion_list]).reset_index(drop = True)     
        # print(gene_weight_up_degree_df)
        for up_row in gene_up_weight_edge_df.itertuples(): 
            if list(up_row[1:])[0] in up_num_deletion_list or list(up_row[3:])[0] in up_num_deletion_list:
                up_deletion_list.append(up_row[0])
        nodefilter_gene_up_weight_edge_df = gene_up_weight_edge_df.drop(gene_up_weight_edge_df.index[up_deletion_list]).reset_index(drop = True)     
        nodefilter_gene_up_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/nodefilter_gene_up_weight_edge.csv', index = False, header = True)
        # DELETE DOWN EDGES SMALLER THAN CERTAIN THRESHOLD
        for down_row in gene_weight_down_degree_df.itertuples():
            if list(down_row[5:])[0] < degree_threshold :
                down_deletion_list.append(down_row[0])
                down_num_deletion_list.append(down_row[1:][0])
        gene_weight_down_degree_df = gene_weight_down_degree_df.drop(gene_weight_down_degree_df.index[down_deletion_list]).reset_index(drop = True)     
        # print(gene_weight_down_degree_df)
        for down_row in gene_down_weight_edge_df.itertuples(): 
            if list(down_row[1:])[0] in down_num_deletion_list or list(down_row[3:])[0] in down_num_deletion_list:
                down_deletion_list.append(down_row[0])
        nodefilter_gene_down_weight_edge_df = gene_down_weight_edge_df.drop(gene_down_weight_edge_df.index[down_deletion_list]).reset_index(drop = True)     
        nodefilter_gene_down_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/nodefilter_gene_down_weight_edge.csv', index = False, header = True)
        # DELETE BIND EDGES SMALLER THAN CERTAIN THRESHOLD
        for bind_row in gene_weight_bind_degree_df.itertuples():
            if list(bind_row[5:])[0] < degree_threshold :
                bind_deletion_list.append(bind_row[0])
                bind_num_deletion_list.append(bind_row[1:][0])
        gene_weight_bind_degree_df = gene_weight_bind_degree_df.drop(gene_weight_bind_degree_df.index[bind_deletion_list]).reset_index(drop = True)     
        # print(gene_weight_bind_degree_df)
        for bind_row in gene_bind_weight_edge_df.itertuples(): 
            if list(bind_row[1:])[0] in bind_num_deletion_list or list(bind_row[3:])[0] in bind_num_deletion_list:
                bind_deletion_list.append(bind_row[0])
        nodefilter_gene_bind_weight_edge_df = gene_bind_weight_edge_df.drop(gene_bind_weight_edge_df.index[bind_deletion_list]).reset_index(drop = True)     
        nodefilter_gene_bind_weight_edge_df.to_csv('.' + dir_opt + '/bianalyse_data/nodefilter_gene_bind_weight_edge.csv', index = False, header = True)



if __name__ == "__main__":
    # BASICAL PARAMETERS IN FILES
    dir_opt = '/datainfo2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'

    ###########################################################################################
    ############### ANALYSE [MSE_LOSS/PEARSON CORRELATION] FROM RECORDED FILES ################
    ###########################################################################################
    path = './datainfo2/result/webgnn_decoder/epoch_75_4'
    epoch_num = 75
    min_train_id = PlotMSECorr(dir_opt).rebuild_loss_pearson(path, epoch_num)
    PlotMSECorr(dir_opt).plot_loss_pearson(path, epoch_num)

    # # ANALYSE DRUG EFFECT
    # print('ANALYSING DRUG EFFECT...')
    epoch_time = '75'
    best_model_num = str(min_train_id)
    PlotMSECorr(dir_opt).plot_train_real_pred(path, best_model_num, epoch_time)
    PlotMSECorr(dir_opt).plot_test_real_pred(path, epoch_time)


    # ###########################################################################################
    # ########################### REBUILD MODEL AND ANALYSIS PARAMTERS ##########################
    # ###########################################################################################
    # dataset_num_list = ['dataset1', 'dataset3', 'dataset5', 'dataset7', 'dataset9']
    # # dataset_num_list = ['dataset1']
    # # dataset_num_list = ['dataset2', 'dataset4', 'dataset6', 'dataset8', 'dataset10']
    # conv_concat = True
    # conv_up_weight_adj_bind, conv_down_weight_adj_bind, conv_mean_weight_adj_bind = \
    #     ReformWeightAdj(dir_opt).form_weight_edge(RNA_seq_filename, dataset_num_list, conv_concat)
    # ReformWeightAdj(dir_opt).form_node_degree(conv_up_weight_adj_bind, conv_down_weight_adj_bind, conv_mean_weight_adj_bind)


    # ###########################################################################################
    # ################################# FILTER EDGES OR NODES ###################################
    # ###########################################################################################
    # edge_threshold = 0.5
    # ReformWeightAdj(dir_opt).filter_edge(edge_threshold)
    # degree_threshold = 10
    # ReformWeightAdj(dir_opt).filter_gene(degree_threshold)