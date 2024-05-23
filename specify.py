import os
import pdb
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from statistics import mean
from statsmodels.distributions.empirical_distribution import ECDF

class Specify():
    def __init__(self):
        pass

    def cancer_cellline_specific(self, top_k, cellline_name):
        # ALL TEST RESULTS
        dl_input_df = pd.read_csv('./data-nci/filtered_data/random_final_dl_input.csv')
        cancer_specific_input_indexlist = []
        for row in dl_input_df.itertuples():
            if row[3] == cellline_name:
                cancer_specific_input_indexlist.append(row[0])
        # CONCAT ALL RESULT IN EACH [TestInput.txt]
        k = 5
        test_pred_df_list = []
        for place_num in range(1, k + 1):
            if place_num == 1:
                result_path = './data-nci/result/webgnn/epoch_500'
            else:
                result_path = './data-nci/result/webgnn/epoch_500_' + str(place_num - 1)
            test_pred_path = result_path + '/BestTestPred.txt'
            test_pred_df = pd.read_table(test_pred_path, delimiter=',')
            test_pred_df_list.append(test_pred_df)
        all_test_pred_df = pd.concat(test_pred_df_list, ignore_index=True)
        # CALCULATE [MSE Loss / Pearson Correlation]
        print('\n------------------------ TEST ' + cellline_name + '  ------------------------\n')
        cancer_specific_test_loss_list = []
        for row in all_test_pred_df.itertuples():
            if row[0] in cancer_specific_input_indexlist:
                mse_loss = (float(row[1]) - float(row[2]))**2
                cancer_specific_test_loss_list.append(mse_loss)
        print('----- MEAN TEST MES LOSS OF ' + cellline_name + ' -----')
        print(mean(cancer_specific_test_loss_list))
        cancer_specific_testpred_df = all_test_pred_df.loc[all_test_pred_df.index.isin(cancer_specific_input_indexlist)]
        cancer_specific_test_pearson = cancer_specific_testpred_df.corr(method = 'pearson')
        print('----- MEAN TEST PEARSON CORRELATION OF ' + cellline_name + ' -----')
        print(cancer_specific_test_pearson)
        # CONCATENATE [DrugA, DrugB, Cell Line Name, Score] + [Score, Pred Score, Test MSE Loss]
        cancer_specific_test_predscore = list(cancer_specific_testpred_df['Pred Score'])
        cancer_specific_test_input_df = dl_input_df.loc[dl_input_df.index.isin(cancer_specific_input_indexlist)]
        cancer_specific_test_input_df.insert(4, 'Pred Score', cancer_specific_test_predscore)
        cancer_specific_test_input_df.insert(5, 'Test MSE Loss', cancer_specific_test_loss_list)
        # GET TEST [TOP PRED SCORE + MINIMUM MSE LOSS] AND DRUGS PAIR
        print('----- TOP ' + str(top_k) + ' TEST SCORE OF ' + cellline_name + ' -----')
        cancer_specific_test_input_df \
            = cancer_specific_test_input_df.sort_values(by=['Score'], ascending=False)
        cancer_specific_top_testinput_df = cancer_specific_test_input_df.head(top_k)
        print(cancer_specific_top_testinput_df)
        cancer_specific_topmin_testinput_df \
            = cancer_specific_top_testinput_df.sort_values(by=['Test MSE Loss'], ascending=True)
        print('----- TOP MINIMUM TEST MES LOSS OF ' + cellline_name + ' -----')
        print(cancer_specific_topmin_testinput_df)
        testloss_topminidx = cancer_specific_topmin_testinput_df['Test MSE Loss'].idxmin()
        testloss_topminobj = cancer_specific_test_input_df.loc[testloss_topminidx, :]
        testloss_topminobj_list = []
        for idx in cancer_specific_topmin_testinput_df.index:
            tmp_obj = cancer_specific_test_input_df.loc[idx, :]
            testloss_topminobj_list.append(tmp_obj)

        # GET TEST [BOTTOM PRED SCORE + MINIMUM MSE LOSS] AND DRUGS PAIR
        print('----- BOTTOM ' + str(top_k) + ' TEST SCORE OF ' + cellline_name + ' -----')
        cancer_specific_test_input_df \
            = cancer_specific_test_input_df.sort_values(by=['Score'], ascending=True)
        cancer_specific_bottom_testinput_df = cancer_specific_test_input_df.head(top_k)
        print(cancer_specific_bottom_testinput_df)
        cancer_specific_bottommin_testinput_df \
            = cancer_specific_bottom_testinput_df.sort_values(by=['Test MSE Loss'], ascending=True)
        print('----- BOTTOM MINIMUM TEST MES LOSS OF ' + cellline_name + ' -----')
        print(cancer_specific_bottommin_testinput_df)
        testloss_bottomminidx = cancer_specific_bottommin_testinput_df['Test MSE Loss'].idxmin()
        testloss_bottomminobj = cancer_specific_test_input_df.loc[testloss_bottomminidx, :]
        testloss_bottomminobj_list = []
        for idx in cancer_specific_bottommin_testinput_df.index:
            tmp_obj = cancer_specific_test_input_df.loc[idx, :]
            testloss_bottomminobj_list.append(tmp_obj)

        return testloss_topminobj_list, testloss_bottomminobj_list

    def cancer_cellline_plot(self, top_k, cellline_name, testloss_obj, filename):
        # ALL TEST RESULTS
        dl_input_df = pd.read_table('./datainfo2/filtered_data/RandomFinalDeepLearningInput.txt', delimiter=',')
        cancer_specific_input_indexlist = []
        for row in dl_input_df.itertuples():
            if row[3] == cellline_name:
                cancer_specific_input_indexlist.append(row[0])
        # CONCAT ALL RESULT IN EACH [TestInput.txt]
        k = 5
        test_pred_df_list = []
        for place_num in range(1, k + 1):
            if place_num == 1:
                result_path = './datainfo2/result/webgnn_decoder/epoch_75'
            else:
                result_path = './datainfo2/result/webgnn_decoder/epoch_75_' + str(place_num - 1)
            test_pred_path = result_path + '/TestPred.txt'
            test_pred_df = pd.read_table(test_pred_path, delimiter=',')
            test_pred_df_list.append(test_pred_df)
        all_test_pred_df = pd.concat(test_pred_df_list, ignore_index=True)
        # CALCULATE [MSE Loss / Pearson Correlation]
        print('\n------------------------ TEST ' + cellline_name + '  ------------------------\n')
        cancer_specific_test_loss_list = []
        for row in all_test_pred_df.itertuples():
            if row[0] in cancer_specific_input_indexlist:
                mse_loss = (float(row[1]) - float(row[2]))**2
                cancer_specific_test_loss_list.append(mse_loss)
        print('----- MEAN TEST MES LOSS OF ' + cellline_name + ' -----')
        print(mean(cancer_specific_test_loss_list))
        cancer_specific_testpred_df = all_test_pred_df.loc[all_test_pred_df.index.isin(cancer_specific_input_indexlist)]
        cancer_specific_test_pearson = cancer_specific_testpred_df.corr(method = 'pearson')
        print('----- MEAN TEST PEARSON CORRELATION OF ' + cellline_name + ' -----')
        print(cancer_specific_test_pearson)
        # CONCATENATE [DrugA, DrugB, Cell Line Name, Score] + [Score, Pred Score, Test MSE Loss]
        cancer_specific_test_predscore = list(cancer_specific_testpred_df['Pred Score'])
        cancer_specific_test_input_df = dl_input_df.loc[dl_input_df.index.isin(cancer_specific_input_indexlist)]
        cancer_specific_test_input_df.insert(4, 'Pred Score', cancer_specific_test_predscore)
        cancer_specific_test_input_df.insert(5, 'Test MSE Loss', cancer_specific_test_loss_list)
        # GET TEST [TOP PRED SCORE + MINIMUM MSE LOSS] AND DRUGS PAIR
        cancer_specific_test_input_df \
            = cancer_specific_test_input_df.sort_values(by=['Score'], ascending=False)

        # PLOT THE KDE FIGURES
        test_score_list = list(cancer_specific_test_input_df['Score'])
        test_score = np.array(test_score_list)
        min_test_score = np.min(test_score)
        max_test_score = np.max(test_score)
        print('Score-min, max: ', min_test_score, max_test_score)
        test_pred_list = list(cancer_specific_test_input_df['Pred Score'])
        test_pred = np.array(test_pred_list)
        min_test_pred = np.min(test_pred)
        max_test_pred = np.max(test_pred)
        print('Pred-min, max: ', min_test_pred, max_test_pred)
        test_drugA = testloss_obj['Drug A']
        test_drugB = testloss_obj['Drug B']
        one_test_score = testloss_obj['Score']
        one_test_pred = testloss_obj['Pred Score']
        test_mseloss = testloss_obj['Test MSE Loss']
        # UNIFY THE LIM RANGE
        xlim_min = np.min([min_test_score, min_test_pred])
        xlim_max = np.max([max_test_score, max_test_pred])

        # Use ECDF to Evaluate the [Score, Pred_score]
        score_ecdf = ECDF(test_score_list)
        score_ecdf_value = score_ecdf(one_test_score)
        print(one_test_score)
        print('Score ECDF: ', score_ecdf_value)
        pred_ecdf = ECDF(test_pred_list)
        pred_ecdf_value = pred_ecdf(one_test_pred)
        print(one_test_pred)
        print('Pred Score ECDF: ', pred_ecdf_value)

        # Easiest Way to Build KDE Plots
        test_score_series = pd.Series(test_score)
        ax_x = test_score_series.plot.kde()
        test_pred_series = pd.Series(test_pred)
        ax_y = test_pred_series.plot(kind='kde')

        # Add notations
        plt.axvline(x=float(one_test_score), c='k', linestyle='--')
        plt.axvline(x=float(one_test_pred), c='r', linestyle='--')
        plt.legend(['Score Distribution', 'Pred Score Distribution', 'Score', 'Pred Score'])
        titlename = test_drugA + ' & ' + test_drugB + ' target on ' + cellline_name + ' cell line'
        plt.title(titlename) 
        plt.savefig(filename, dpi = 600)
        # plt.show()
        plt.close()




if __name__ == "__main__":
    ##############################################################
    ############# LISTS OF CELL LINE SPECIFIC ANALYSIS ###########
    ##############################################################
    top_k = 10
    cellline_name = 'DU-145'
    # cellline_name = 'PC-3'
    testloss_topminobj_list, testloss_bottomminobj_list = Specify().cancer_cellline_specific(top_k, cellline_name)

    # top_n = 1
    # testloss_topminobj = testloss_topminobj_list[top_n - 1]
    # testloss_bottomminobj = testloss_bottomminobj_list[top_n - 1]

    # # MAKE BOXPLOTS
    # topmin_loss = True
    # if cellline_name == 'A549/ATCC':
    #     bind_plot_path = './datainfo2/bianalyse_data/A549' + '/bind_plots'
    # else:
    #     bind_plot_path = './datainfo2/bianalyse_data/' + cellline_name + '/bind_plots'
    # if topmin_loss == True:
    #     if cellline_name == 'A549/ATCC':
    #         filename = bind_plot_path + '/kdeplot_topmin_A549_top_' + str(top_n) + '.png'
    #     else:
    #         filename = bind_plot_path + '/kdeplot_topmin_'  + cellline_name + '_top_' + str(top_n) + '.png'
    #     Specify(dir_opt).cancer_cellline_plot(top_k, cellline_name, testloss_topminobj, filename)
    # else:
    #     if cellline_name == 'A549/ATCC':
    #         filename = bind_plot_path + '/kdeplot_bottommin_A549_bottom_' + str(top_n) + '.png'
    #     else:
    #         filename = bind_plot_path + '/kdeplot_bottommin_'  + cellline_name + '_bottom_' + str(top_n) + '.png'
    #     Specify(dir_opt).cancer_cellline_plot(top_k, cellline_name, testloss_bottomminobj, filename)
