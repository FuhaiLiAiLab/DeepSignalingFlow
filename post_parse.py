import os
import re
import numpy as np
import pandas as pd

from load_data import LoadData

# RANDOMIZE THE [final_NCI60_DeepLearningInput]
def input_random():
    final_input_df = pd.read_csv('./data/filtered_data/final_dl_input.csv')
    random_final_input_df = final_input_df.sample(frac = 1)
    random_final_input_df.to_csv('./data/filtered_data/random_final_dl_input.csv', index = False, header = True)

# SPLIT DEEP LEARNING INPUT INTO TRAINING AND TEST
def split_k_fold(k):
    random_final_dl_input_df = pd.read_csv('./data/filtered_data/random_final_dl_input.csv')
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
        split_input_df.to_csv('./data/filtered_data/split_input_' + str(place_num + 1) + '.csv', index = False, header = True)


# input_random()
# split_k_fold(k=5)


if os.path.exists('./data/form_data') == False:
    os.mkdir('./data/form_data')
batch_size = 64
LoadData().load_all_split(batch_size, k)

k = 5
for n_fold in range(1, k + 1):
    # ############## MOUDLE 2 ################
    print('split_input_' + str(n_fold) + '.csv')
    LoadData().load_adj_edgeindex()

    ################ MOUDLE 3 ################
    # FORM N-TH FOLD TRAINING DATASET
    LoadData().load_train_test(k, n_fold)