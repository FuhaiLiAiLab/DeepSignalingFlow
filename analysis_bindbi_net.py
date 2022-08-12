import os
import pdb
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from parse_file import ParseFile
from parse_file import ParseFile, k_fold_split
from webgnn_dec_analysis import PlotMSECorr
from specify import Specify

class NetAnalyse():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def statistic_net(self):
        # READ INFORMATION FROM FILE
        dir_opt = self.dir_opt
        gene_bind_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_bind_weight_edge.csv')
        gene_bind_degree_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_weight_bind_degree.csv')
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}
        # print(cellline_gene_num_dict)
        # BINPLOT OF WEIGHT DISTRIBUTION
        bind_weight_list = []
        for bind_row in gene_bind_weight_edge_df.itertuples():
            bind_weight_list.append(up_row[5])
        bins = np.arange(-0.5, 3, 0.2) # FIXED BIN SIZE
        plt.xlim([min(bind_weight_list) - 0.2, max(bind_weight_list) + 0.2])
        plt.hist(bind_weight_list, bins = bins)
        plt.title('Gene-Gene Bindflow Weight Distribution')
        plt.ylabel('count')
        plt.show()
        # BINPLOT OF DEGREE DISTRIBUTION
        bind_degree_list = []
        for bind_row in gene_bind_degree_df.itertuples():
            bind_degree_list.append(bind_row[5])
        bins = np.arange(0, 120, 20) # FIXED BIN SIZE
        plt.xlim([min(bind_degree_list) - 5, max(bind_degree_list) + 20])
        plt.hist(bind_degree_list, bins = bins)
        plt.title('Gene Bindflow Degree Distribution')
        plt.ylabel('count')
        plt.show()

    
    def drug_bank_filter(self):
        # READ FINAL [RandomFinalDeepLearningInput] FOR [drugs, genes]
        dir_opt = self.dir_opt
        random_final_dl_input_df = pd.read_table('.' + dir_opt + '/filtered_data/RandomFinalDeepLearningInput.txt', delimiter = ',')
        random_final_drug_list = []
        for drug in random_final_dl_input_df['Drug A']:
            if drug not in random_final_drug_list:
                random_final_drug_list.append(drug)
        for drug in random_final_dl_input_df['Drug B']:
            if drug not in random_final_drug_list:
                random_final_drug_list.append(drug)
        random_final_drug_list = sorted(random_final_drug_list)
        # print(random_final_drug_list)
        # print(len(random_final_drug_list))
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])

        # READ ORIGINAL [drugBank] 
        # [GOAL]: TRANSFER [drug_target_df] TO MAPPED NAMES
        drug_target_df = pd.read_table('.' + dir_opt + '/init_data/drug_tar_drugBank_all.txt')
        # [FIRST]: FORM DRUG DICT ALIGN WITH [RandomFinalDeepLearningInput.txt]
        drug_map_dict = np.load('.' + dir_opt + '/filtered_data/drug_map_dict.npy', allow_pickle='TRUE').item()
        drug_deletion_list = []
        for key in drug_map_dict.keys():
            if key not in random_final_drug_list:
                drug_deletion_list.append(key)
        filtered_drug_map_dict = drug_map_dict
        [filtered_drug_map_dict.pop(key) for key in drug_deletion_list]
        for values in filtered_drug_map_dict.values():
            if values is None:
                print('NEED TO HANDLE NONE')
        # print(filtered_drug_map_dict) # [DL : DrugBank]

        # [SECOND]: FORM TARGET DICT ALIGN WITH [RandomFinalDeepLearningInput.txt]
        drug_dict, drug_num_dict, target_dict, target_num_dict = ParseFile(dir_opt).drug_target()
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        # print(target_dict)
        gene_target_name_dict = {}
        count = 0
        for row in cellline_gene_df.itertuples():
            if row[2] not in target_dict.keys(): 
                map_index = -1
                gene_target_name_dict[row[2]] = map_index
            else:
                map_index = target_dict[row[2]]
                gene_target_name_dict[row[2]] = target_num_dict[map_index]
            count +=1
        # print(gene_target_name_dict) # [DL : DrugBank]

        # SUBSTITUE ALL NAMES [Drug, Target] TO DL NAMES
        if os.path.exists('.' + dir_opt + '/post_data') == False:
            os.mkdir('.' + dir_opt + '/post_data')
        # [Drugs]
        new_drug_target_df1 = drug_target_df.replace(filtered_drug_map_dict.values(), filtered_drug_map_dict.keys())
        new_drug_target_df1.to_csv('.' + dir_opt + '/post_data/new_drug_target_df1.txt', index = False, header = True)
        # [Targets]
        new_drug_target_df2 = new_drug_target_df1.replace(gene_target_name_dict.values(), gene_target_name_dict.keys())
        new_drug_target_df2.to_csv('.' + dir_opt + '/post_data/new_drug_target_df2.txt', index = False, header = True)
        # DELETE NOT EXISTED [Drugs, Targets] IN [RandomFinalDeepLearningInput.txt]
        row_deletion_list = []
        for row in new_drug_target_df2.itertuples():
            if row[1] not in random_final_drug_list or row[2] not in cellline_gene_list : 
                row_deletion_list.append(row[0])
        # print(row_deletion_list)
        filtered_drug_target_df = new_drug_target_df2.drop(new_drug_target_df2.index[row_deletion_list]).reset_index(drop = True)
        filtered_drug_target_df = filtered_drug_target_df.sort_values(by = ['Drug']).reset_index(drop = True)
        # print(filtered_drug_target_df)
        filtered_drug_target_df.to_csv('.' + dir_opt + '/post_data/filtered_drug_target_df.txt', index = False, header = True)

        # FORM [gene_targeted_drug_dict]
        gene_targeted_drug_dict = {}
        gene_targeted_list = []
        filtered_target_list = list(filtered_drug_target_df['Target'])
        for gene in cellline_gene_list:
            if gene in filtered_target_list:
                gene_targeted_list.append(gene)
                temp_gene_targeted_drug_list = list(filtered_drug_target_df.loc[filtered_drug_target_df['Target'] == gene]['Drug'])
                gene_targeted_drug_dict[gene] = temp_gene_targeted_drug_list
        print(gene_targeted_drug_dict)
        np.save('.' + dir_opt + '/post_data/gene_targeted_drug_dict.npy', gene_targeted_drug_dict)
        return gene_targeted_list
        

    def plot_target_gene(self, node_threshold, gene_targeted_list):
        # READ INFORMATION FROM FILE
        dir_opt = self.dir_opt
        gene_bind_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_bind_weight_edge.csv')
        gene_bind_degree_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_weight_bind_degree.csv')
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}
        cellline_gene_name_dict = {cellline_gene_list[i - 1] : i for i in range(1, len(cellline_gene_list) + 1)}
        # BUILD bind NODES DEGREE [DELETION LIST]
        filter_node_count = 0
        bind_node_deletion_index_list = []
        bind_node_index_list = []
        bind_node_name_list = []
        bind_node_degree_list = []
        for bind_row in gene_bind_degree_df.itertuples():
            if list(bind_row[5:])[0] <= node_threshold or bind_row[1] == 1572:
                bind_node_deletion_index_list.append(bind_row[1])
            else:
                filter_node_count += 1
                bind_node_name_list.append(bind_row[2])
                bind_node_index_list.append(bind_row[1])
                bind_node_degree_list.append(bind_row[5])
        # print('---------- GENE TARGETRED BY DRUG LIST ------------')
        # print(gene_targeted_list)
        # print(bind_node_name_list)
        # INTERSECT ON [gene_targeted_set](TARGETED NODES) [bind_node_name_set](FILTERED BY THRESHOLD NODES)
        gene_targeted_set = set(gene_targeted_list)
        bind_node_name_set = set(bind_node_name_list)
        set1 = gene_targeted_set.intersection(bind_node_name_set)
        intersection_name_list = list(set1)
        intersection_list = [cellline_gene_name_dict[item] for item in intersection_name_list]

        intersection_degree_list = []
        for gene in intersection_list:
            gene_degree = float(gene_bind_degree_df.loc[gene_bind_degree_df['gene_idx'] == gene]['degree'])
            intersection_degree_list.append(gene_degree)
        print('----- GENE INTERSECTED TARGETRED BY DRUG LIST: ' + str(len(intersection_name_list)) + ' -----')
        print(intersection_name_list)
        return intersection_name_list, intersection_list, intersection_degree_list



    def drug_target_interaction(self, cellline_name, topmin_loss, testloss_topminobj, testloss_bottomminobj):
        filtered_drug_target_df = pd.read_csv('./datainfo2/post_data/filtered_drug_target_df.txt', delimiter=',')
        print(filtered_drug_target_df)
        # FORM [rna_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1 INSTEAD OF 0 !!!
        rna_gene_df = pd.read_csv('./datainfo2/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        rna_gene_list = list(rna_gene_df['geneSymbol'])
        rna_gene_dict = {rna_gene_list[i - 1] : i for i in range(1, len(rna_gene_list) + 1)}
        rna_gene_dict_df = pd.DataFrame({'rna_gene': list(rna_gene_dict.keys()), 'gene_num': list(rna_gene_dict.values())})
        rna_gene_dict_df.to_csv('./datainfo2/filtered_data/gene_num_dict.csv', index=False, header=True)
        # FORM [dl_input_drug_dict] TO MAP GENES WITH INDEX START FROM 1585
        drug_map_df = pd.read_csv('./datainfo2/filtered_data/dlinput_drugbank_num_dict.csv')
        drug_map_num_list = list(drug_map_df['drug_index'])
        dl_druglist = list(drug_map_df['dl_input'])
        drugbank_druglist = list(drug_map_df['drugbank'])
        dl_drugbank_dict = {dl_druglist[i - 1] : drugbank_druglist[i - 1] for i in range(1, len(drugbank_druglist)+1)}
        drugbank_drug_dict = {drugbank_druglist[i - 1] : drug_map_num_list[i - 1] for i in range(1, len(drugbank_druglist)+1)}

        print('\n-------- TEST ' + cellline_name + ' --------\n')

        if topmin_loss == True:
            # These 2 drug names are from [dl_input]
            # But [filtered_drug_target_df]'s Drug Names == dl_Drug Names
            top_testloss_dl_drugA = testloss_topminobj['Drug A']
            top_testloss_dl_drugB = testloss_topminobj['Drug B']
            print(top_testloss_dl_drugA, top_testloss_dl_drugB)
            # Convert these 2 drug names into [drugbank]
            top_testloss_drugA = dl_drugbank_dict[top_testloss_dl_drugA]
            top_testloss_drugB = dl_drugbank_dict[top_testloss_dl_drugB]
            print(top_testloss_drugA, drugbank_drug_dict[top_testloss_drugA])
            print(top_testloss_drugB, drugbank_drug_dict[top_testloss_drugB])
            # Get Cell Line Specific Drug Target Links
            cellline_specific_drugtar_list = []
            for row in filtered_drug_target_df.itertuples():
                if row[1] == top_testloss_dl_drugA or row[1] == top_testloss_dl_drugB:
                    print(row)
                    cellline_specific_drugtar_list.append(row[0])
            drugbank_num_df = pd.read_csv('./datainfo2/form_data/drugbank_num.txt', delimiter=',')
            cellline_specific_drugbank_df = drugbank_num_df.loc[drugbank_num_df.index.isin(cellline_specific_drugtar_list)]
            print(cellline_specific_drugbank_df)
            print(testloss_topminobj)
            return cellline_specific_drugbank_df

        else:
            # These 2 drug names are from [dl_input]
            # But [filtered_drug_target_df]'s Drug Names == dl_Drug Names
            bottom_testloss_dl_drugA = testloss_bottomminobj['Drug A']
            bottom_testloss_dl_drugB = testloss_bottomminobj['Drug B']
            print(bottom_testloss_dl_drugA, bottom_testloss_dl_drugB)
            # Convert these 2 drug names into [drugbank]
            bottom_testloss_drugA = dl_drugbank_dict[bottom_testloss_dl_drugA]
            bottom_testloss_drugB = dl_drugbank_dict[bottom_testloss_dl_drugB]
            print(bottom_testloss_drugA, drugbank_drug_dict[bottom_testloss_drugA])
            print(bottom_testloss_drugB, drugbank_drug_dict[bottom_testloss_drugB])
            # Get Cell Line Specific Drug Target Links
            cellline_specific_drugtar_list = []
            for row in filtered_drug_target_df.itertuples():
                if row[1] == bottom_testloss_dl_drugA or row[1] == bottom_testloss_dl_drugB:
                    print(row)
                    cellline_specific_drugtar_list.append(row[0])
            drugbank_num_df = pd.read_csv('./datainfo2/form_data/drugbank_num.txt', delimiter=',')
            cellline_specific_drugbank_df = drugbank_num_df.loc[drugbank_num_df.index.isin(cellline_specific_drugtar_list)]
            print(cellline_specific_drugbank_df)
            print(testloss_bottomminobj)
            return cellline_specific_drugbank_df



    def plot_net2(self, node_threshold, edge_threshold, 
                intersection_list, intersection_degree_list, cellline_specific_drugbank_df,
                topmin_loss, seed, cellline_name, top_n):
        # READ INFORMATION FROM FILE
        dir_opt = self.dir_opt
        gene_bind_weight_edge_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_bind_weight_edge.csv')
        gene_bind_degree_df = pd.read_csv('.' + dir_opt + '/bianalyse_data/gene_weight_bind_degree.csv')
        # FORM [cellline_gene_dict] TO MAP GENES WITH INDEX NUM !!! START FROM 1
        cellline_gene_df = pd.read_csv('.' + dir_opt + '/filtered_data/nci60-ccle_RNAseq_tpm2.csv')
        cellline_gene_list = list(cellline_gene_df['geneSymbol'])
        cellline_gene_num_dict = {i : cellline_gene_list[i - 1] for i in range(1, len(cellline_gene_list) + 1)}

        # BUILD bind NODES DEGREE [DELETION LIST]
        filter_node_count = 0
        bind_node_deletion_index_list = []
        bind_node_index_list = []
        bind_node_name_list = []
        bind_node_degree_dict = {}
        for bind_row in gene_bind_degree_df.itertuples():
            if list(bind_row[5:])[0] <= node_threshold:
                bind_node_deletion_index_list.append(bind_row[1])
            else:
                filter_node_count += 1
                bind_node_name_list.append(bind_row[2])
                bind_node_index_list.append(bind_row[1])
                bind_node_degree_dict[bind_row[1]] = bind_row[5]
        print('----- FILTERED GENES WITH HIGHER DEGREES: ' + str(filter_node_count) + ' -----')
        # REMOVE CERTAIN DELETED GENES IN [cellline_gene_num_dict]
        bind_filter_gene_dict = cellline_gene_num_dict
        [bind_filter_gene_dict.pop(key) for key in bind_node_deletion_index_list]
        # BUILD bind DIRECTED GRAPH
        bind_digraph = nx.Graph() 
        edge_remove = False # DO NOT REMOVE EDGES, ONLY REMOVE NODES
        bind_edge_deletion_list = []
        for bind_row in gene_bind_weight_edge_df.itertuples():
            if edge_remove == True:
                if bind_row[5] <= edge_threshold:
                    bind_edge_deletion_list.append((bind_row[1], bind_row[3]))
            bind_digraph.add_edge(bind_row[1], bind_row[3], weight = bind_row[5])

        # BUILD [drug target]
        node_degree_max = max(list(bind_node_degree_dict.values()))
        edge_weight_max = max(list(gene_bind_weight_edge_df['weight']))
        for row in cellline_specific_drugbank_df.itertuples():
            drug_idx = row[1]
            gene_idx = row[2]
            # degree
            bind_node_degree_dict[drug_idx] = 0.8 * node_degree_max
            # weight
            bind_digraph.add_edge(drug_idx, gene_idx, weight = 0.5 * edge_weight_max)
        # Show [drugbank]'s Drug Names in Final Graph
        drug_map_df = pd.read_csv('./datainfo2/filtered_data/dlinput_drugbank_num_dict.csv')
        drug_map_num_list = list(drug_map_df['drug_index'])
        drugbank_druglist = list(drug_map_df['drugbank'])
        drugbank_num_drug_dict = {drug_map_num_list[i - 1] : drugbank_druglist[i - 1] for i in range(1, len(drugbank_druglist)+1)}
        cellline_specific_druglist = list(set(list(cellline_specific_drugbank_df['Drug'])))
        bind_filter_node_name_dict = bind_filter_gene_dict
        for drug_idx in cellline_specific_druglist:
            drugbank_name = drugbank_num_drug_dict[drug_idx]
            # label
            bind_filter_node_name_dict[drug_idx] = drugbank_name
        
        
        # RESTRICT GRAPH WITH NODES DEGREES BY USING [nx.restricted_view]
        if edge_remove == True:
            bind_filtered_digraph = nx.restricted_view(bind_digraph, nodes = bind_node_deletion_index_list, edges = bind_edge_deletion_list)
        else:
            bind_filtered_digraph = nx.restricted_view(bind_digraph, nodes = bind_node_deletion_index_list, edges = [])
        print('----- FILTERED GRAPH\'s EDGES WITH HIGHER NODE DEGREES: ' + str(len(bind_filtered_digraph.edges())) + ' -----')
        edges = bind_filtered_digraph.edges()
        weights = [bind_filtered_digraph[u][v]['weight'] for u, v in edges]

        # import pdb; pdb.set_trace()  

        nodes = bind_filtered_digraph.nodes()
        bind_filtered_degree_list = []
        for node_idx in nodes:
            node_degree = bind_node_degree_dict[node_idx]
            bind_filtered_degree_list.append(node_degree)
        
        bind_node_degree_list = bind_filtered_degree_list


        # FIND Critical Nodes
        cellline_specific_targetlist = list(set(list(cellline_specific_drugbank_df['Target'])))
        bind_graph_target_nodes = []
        for node in nodes:
            if node in cellline_specific_targetlist:
                bind_graph_target_nodes.append(node)
        print('----- FILTERED GRAPH\'s DRUG TARGET NODES -----')
        print(bind_graph_target_nodes)
        # FIND Critical Edges
        bind_graph_drugtar_links = []
        for u, v in edges:
            if v in cellline_specific_targetlist or u in cellline_specific_targetlist:
                if u <= 1584 and v <= 1584:
                    continue
                else:
                    bind_graph_drugtar_links.append((u, v))
        print('----- FILTERED GRAPH\'s DRUG TARGET LINKS -----')
        print(bind_graph_drugtar_links)

        # import pdb; pdb.set_trace()

        # FIND Critical Paths
        cellline_specific_druglist = list(set(list(cellline_specific_drugbank_df['Drug'])))
        drug_1 = cellline_specific_druglist[0]
        drug_2 = cellline_specific_druglist[1]
        # import pdb; pdb.set_trace()
        # cutoff==3
        cutoff = 3
        path3_num_count = 0
        bind_graph_target_between_3links = []
        path3 = nx.all_simple_paths(bind_filtered_digraph, source=drug_1, target=drug_2, cutoff=3)
        path3_lists = list(path3)
        for path in path3_lists:
            if len(path) < 3:
                print(path)
                continue
            path3_num_count += 1
            for i in range(1, cutoff):
                node_u = path[i - 1]
                node_v = path[i]
                bind_graph_target_between_3links.append((node_u, node_v))
        bind_graph_target_between_3links = list(set(bind_graph_target_between_3links))
        print('----- DRUG BETWEENESS LINKS OF PATH 3: -----')
        print(path3_lists)
        print(bind_graph_target_between_3links)
        print('----- NUMBERS OF PATH 3: -----')
        print(path3_num_count)
        # cutoff==4
        path4_num_count = 0
        path4_weight_count = 0
        cutoff = 4
        bind_graph_target_between_4links = []
        path4 = nx.all_simple_paths(bind_filtered_digraph, source=drug_1, target=drug_2, cutoff=4)
        path4_lists = list(path4)
        # import pdb; pdb.set_trace()
        for path in path4_lists:
            if len(path) < 5:
                continue
            path4_num_count += 1
            for i in range(1, cutoff):
                node_u = path[i - 1]
                node_v = path[i]
                bind_graph_target_between_4links.append((node_u, node_v))
                this_weight = bind_filtered_digraph[node_u][node_v]['weight']
                path4_weight_count += this_weight
        bind_graph_target_between_4links = list(set(bind_graph_target_between_4links))
        print('----- DRUG BETWEENESS LINKS OF PATH 4: -----')
        print(path4_lists)
        print(bind_graph_target_between_4links)
        print('----- NUMBERS OF PATH 4: -----')
        print(path4_num_count)
        print('----- SUM OF WEIGHTS FOR PATH 4: -----')
        print(path4_weight_count)

        if topmin_loss == False:
            # cutoff==5
            cutoff = 5
            path5_num_count = 0
            bind_graph_target_between_5links = []
            path5 = nx.all_simple_paths(bind_filtered_digraph, source=drug_1, target=drug_2, cutoff=5)
            path5_lists = list(path5)
            # path_lists = path_lists[0:3]
            # print(path5_lists)
            for path in path5_lists:
                if path in path4_lists:
                    continue
                path5_num_count += 1
                for i in range(1, cutoff):
                    node_u = path[i - 1]
                    node_v = path[i]
                    bind_graph_target_between_5links.append((node_u, node_v))
            bind_graph_target_between_5links = list(set(bind_graph_target_between_5links))
            print('----- DRUG BETWEENESS LINKS OF PATH 5: -----')
            print(bind_graph_target_between_5links)
            print('----- NUMBERS OF PATH 5: -----')
            print(path5_num_count)

        # DRAW GRAPHS WITH CERTAIN TYPE
        pos = nx.spring_layout(bind_filtered_digraph, 
                                k = 15.0, 
                                center=(1,20),
                                scale = 10, 
                                iterations = 2000,
                                seed = seed)

        cmap = plt.cm.viridis
        dmin = min(bind_node_degree_list)
        dmax = max(bind_node_degree_list)
        cmap2 = plt.cm.OrRd
        emin = min(weights)
        emax = max(weights)
        fig=plt.figure(figsize=(10, 6)) 

        nx.draw_networkx_edges(bind_filtered_digraph, 
                    pos = pos,
                    # arrowsize = 5,
                    alpha = 0.3,
                    width = [float(weight**3) / 1 for weight in weights],
                    # edge_color = rerange_weights,
                    edge_color = weights,
                    edge_cmap = cmap2
                    )

        # HIGHLIGHT NODES WITH TARGET (effect = circle those critical nodes)
        hilight_nodes = nx.draw_networkx_nodes(bind_filtered_digraph.subgraph(intersection_list), 
                    pos = pos,
                    nodelist = intersection_list, 
                    node_size = [float(degree**1.75) / 10 for degree in intersection_degree_list],
                    linewidths = 1.0,
                    node_color = 'white'
                    )
        hilight_nodes.set_edgecolor('red')
        # HIGHLIGHT NODES WITH DRUG TARGET ONLY IN THIS SUBNET (effect = circle those critical nodes)
        hilight_genes = nx.draw_networkx_nodes(bind_filtered_digraph.subgraph(bind_graph_target_nodes), 
                    pos = pos,
                    nodelist = bind_graph_target_nodes, 
                    node_size = [float(bind_node_degree_dict[gene_idx]**1.75) / 10 for gene_idx in bind_graph_target_nodes],
                    linewidths = 1.0,
                    node_color = 'white'
                    )
        hilight_genes.set_edgecolor('blue')

        nx.draw_networkx_nodes(bind_filtered_digraph, 
                    pos = pos,
                    node_color = bind_node_degree_list,
                    nodelist = nodes, 
                    node_size = [float(degree**1.75) / 20 for degree in bind_node_degree_list],
                    alpha = 0.9,
                    cmap = cmap
                    )

        # HIGHLIGHT DRUUGS
        # First remove nodes in original color
        hilight_drugs = nx.draw_networkx_nodes(bind_filtered_digraph.subgraph(cellline_specific_druglist), 
                    pos = pos,
                    nodelist = cellline_specific_druglist, 
                    node_size = [float(bind_node_degree_dict[drug_idx]**1.75) / 20 for drug_idx in cellline_specific_druglist],
                    linewidths = 1.5,
                    alpha = 1,
                    node_color = 'white',
                    )
        # Shape = "V" And Plot
        hilight_drugs = nx.draw_networkx_nodes(bind_filtered_digraph.subgraph(cellline_specific_druglist), 
                    pos = pos,
                    nodelist = cellline_specific_druglist, 
                    node_size = [float(bind_node_degree_dict[drug_idx]**1.75) / 5.0 for drug_idx in cellline_specific_druglist],
                    linewidths = 0.5,
                    alpha = 1,
                    node_color = 'orange',
                    node_shape = 'v'
                    )
        hilight_drugs.set_edgecolor('black')


        # HIGHLIGHT [drug-target-betweeness] EDGES (cutoff==4)
        nx.draw_networkx_edges(bind_filtered_digraph,
                    pos = pos,
                    edgelist = bind_graph_target_between_4links,
                    connectionstyle = 'arc3, rad = 0.3',
                    edge_color = 'lightgreen',
                    width = 0.9)
        
        # HIGHLIGHT [drug-target-betweeness] EDGES (cutoff==3)
        nx.draw_networkx_edges(bind_filtered_digraph,
                    pos = pos,
                    edgelist = bind_graph_target_between_3links,
                    connectionstyle = 'arc3, rad = 0.3',
                    edge_color = 'red',
                    width = 1.5)
        
        # HIGHLIGHT [drug-target-betweeness] EDGES (cutoff==5)
        if topmin_loss == False:
            nx.draw_networkx_edges(bind_filtered_digraph,
                        pos = pos,
                        edgelist = bind_graph_target_between_5links,
                        connectionstyle = 'arc3, rad = 0.3',
                        edge_color = 'orange',
                        width = 0.5)

        # HIGHLIGHT [drug-target] EDGES
        nx.draw_networkx_edges(bind_filtered_digraph,
                    pos = pos,
                    edgelist = bind_graph_drugtar_links,
                    connectionstyle = 'arc3, rad = 0.3',
                    edge_color = 'lightblue',
                    width = 1.5)

        nx.draw_networkx_labels(bind_filtered_digraph,
                    pos = pos,
                    labels = bind_filter_node_name_dict,
                    font_size = 2.0
                    )

        # import pdb; pdb.set_trace()
        degree = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = dmin, vmax = dmax))
        dcl = plt.colorbar(degree)
        dcl.ax.tick_params(labelsize = 8)
        dcl.ax.set_ylabel('Nodes Degree')
        edge = plt.cm.ScalarMappable(cmap = cmap2, norm = plt.Normalize(vmin = emin, vmax = emax))
        ecl = plt.colorbar(edge)
        ecl.ax.tick_params(labelsize = 8)
        ecl.ax.set_ylabel('Edges Weight')
        
        drugA = drugbank_num_drug_dict[cellline_specific_druglist[0]]
        drugB = drugbank_num_drug_dict[cellline_specific_druglist[1]]
        titlename = drugA + ' & ' + drugB + ' target on filtered ' + str(filter_node_count) \
            + ' genes on ' + cellline_name + ' cell line'
        plt.title(titlename) 
        # plt.show()
        if cellline_name == 'A549/ATCC':
            cellline_name = 'A549'
        bind_plot_path = './datainfo2/bianalyse_data/' + cellline_name + '/bind_plots'
        if os.path.exists(bind_plot_path) == False:
            os.mkdir(bind_plot_path)
        if topmin_loss == True:
            filename = bind_plot_path + '/plot_topmin_'  + cellline_name + '_top_' + str(top_n) + '.png'
        else:
            filename = bind_plot_path + '/plot_bottommin_'  + cellline_name + '_bottom_' + str(top_n) + '.png'
        plt.savefig(filename, dpi = 600)
        plt.close()




if __name__ == "__main__":
    # BASICAL PARAMETERS IN FILES
    dir_opt = '/datainfo2'
    # NetAnalyse(dir_opt).statistic_net()

    node_threshold = 10.0
    edge_threshold = 1.0

    gene_targeted_list = NetAnalyse(dir_opt).drug_bank_filter()
    # print(gene_targeted_list)

    intersection_name_list, intersection_list, intersection_degree_list = \
        NetAnalyse(dir_opt).plot_target_gene(node_threshold, gene_targeted_list)


    # SET TRAINING/TEST SET
    top_k = 10
    seed = 187
    topmin_loss = False
    ###### Prostate Cancer ######
    # cellline_name = 'DU-145'
    # cellline_name = 'PC-3'

    ###### Brain Cancer (Central Neural System) ######
    # cellline_name = 'SF-268'
    # cellline_name = 'SF-295'
    # cellline_name = 'SF-539'
    #### cellline_name = 'SNB-19'
    #### cellline_name = 'SNB-75'
    # cellline_name = 'U251'

    ###### Lung Cancer ######
    # cellline_name = 'A549/ATCC'
    # cellline_name = 'EKVX'
    # cellline_name = 'HOP-62'
    # cellline_name = 'HOP-92'
    # cellline_name = 'NCI-H226'
    # cellline_name = 'NCI-H23'
    # cellline_name = 'NCI-H322M'
    cellline_name = 'NCI-H460'
    # cellline_name = 'NCI-H522'

    # PREPARE FOLDER PATH
    dir_opt = '/datainfo2'
    if cellline_name == 'A549/ATCC':
        cellline_plot_path = './datainfo2/bianalyse_data/A549'
    else:
        cellline_plot_path = './datainfo2/bianalyse_data/' + cellline_name
    if os.path.exists(cellline_plot_path) == False:
        os.mkdir(cellline_plot_path)
    # GET TESTLOSS TOP/BOTTOM Object List
    testloss_topminobj_list, testloss_bottomminobj_list = Specify(dir_opt).cancer_cellline_specific(top_k, cellline_name)
    top_n = 5
    testloss_topminobj = testloss_topminobj_list[top_n - 1]
    testloss_bottomminobj = testloss_bottomminobj_list[top_n - 1]    

    # PLOT NETWORKS
    cellline_specific_drugbank_df = NetAnalyse(dir_opt).drug_target_interaction(cellline_name, topmin_loss, testloss_topminobj, testloss_bottomminobj)
    NetAnalyse(dir_opt).plot_net2(node_threshold, edge_threshold,
        intersection_list, intersection_degree_list, cellline_specific_drugbank_df, topmin_loss, seed, cellline_name, top_n)

    # MAKE BOXPLOTS
    if cellline_name == 'A549/ATCC':
        bind_plot_path = './datainfo2/bianalyse_data/A549' + '/bind_plots'
    else:
        bind_plot_path = './datainfo2/bianalyse_data/' + cellline_name + '/bind_plots'
    if topmin_loss == True:
        if cellline_name == 'A549/ATCC':
            filename = bind_plot_path + '/kdeplot_topmin_A549_top_' + str(top_n) + '.png'
        else:
            filename = bind_plot_path + '/kdeplot_topmin_'  + cellline_name + '_top_' + str(top_n) + '.png'
        Specify(dir_opt).cancer_cellline_plot(top_k, cellline_name, testloss_topminobj, filename)
    else:
        if cellline_name == 'A549/ATCC':
            filename = bind_plot_path + '/kdeplot_bottommin_A549_bottom_' + str(top_n) + '.png'
        else:
            filename = bind_plot_path + '/kdeplot_bottommin_'  + cellline_name + '_bottom_' + str(top_n) + '.png'
        Specify(dir_opt).cancer_cellline_plot(top_k, cellline_name, testloss_bottomminobj, filename)
