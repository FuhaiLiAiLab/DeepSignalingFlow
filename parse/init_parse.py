import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def minMax(x): return pd.Series(index=['min','max'],data=[x.min(),x.max()])


class ReadFile():
    def __init__(self, dataset):
        self.dataset = dataset

    def nci_combo_input(self):
        dataset = self.dataset
        ### INTIALIZE [NCI60 DrugScreen Data]
        print('----- READING NCI60 DRUG SCREEN RAW DATA -----')
        if os.path.exists('../' + dataset +  '/init_data') == False:
            os.mkdir('../' + dataset +  '/init_data')
        dl_input_df = pd.read_csv('../' + dataset +  '/raw_data/NCI60/DeepLearningInput.csv')
        dl_input_df = dl_input_df.groupby(['Drug A', 'Drug B', 'Cell Line Name']).agg({'Score':'mean'}).reset_index()
        # REMOVE SINGLE DRUG SCREEN DATA [Actual Fact Shows No Single Drug]
        dl_input_deletion_list = []
        dl_input_df = dl_input_df.fillna('missing')
        for row in dl_input_df.itertuples():
            if row[1] == 'missing' or row[2] == 'missing':
                dl_input_deletion_list.append(row[0])
        dl_input_df = dl_input_df.drop(dl_input_df.index[dl_input_deletion_list]).reset_index(drop=True)
        dl_input_df.to_csv('../' + dataset +  '/init_data/almanac_dl_input.csv', index=False, header=True)
        ### PROFILE [Number of Drugs / Number of Cell Lines]
        drug_list = list(set(list(dl_input_df['Drug A']) + list(dl_input_df['Drug B'])))
        cell_line_list = list(set(list(dl_input_df['Cell Line Name'])))
        print('----- NUMBER OF DRUGS IN NCI ALMANAC: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF CELL LINES IN NCI ALMANAC: ' + str(len(cell_line_list)) + ' -----')
        print(dl_input_df.shape)

    def parse_drugcomb_fi(self):
        dataset = self.dataset
        ### INTIALIZE [O'NEIL DrugScreen Data]
        print('----- READING DrugComb_fi DRUG SCREEN RAW DATA -----')
        if os.path.exists('../' + dataset +  '/init_data') == False:
            os.mkdir('../' + dataset +  '/init_data')
        drugcomb_fi_df = pd.read_csv('../' + dataset +  '/raw_data/drugcomb-fi/summary_v_1_5.csv')
        ### SELECT ONLY [ONEIL] DATASET
        oneil_df = drugcomb_fi_df.loc[drugcomb_fi_df['study_name'] == 'ONEIL']
        oneil_comb_df = oneil_df.loc[oneil_df['drug_col'].notnull()]
        oneil_comb_df['synergy_loewe'] = oneil_comb_df['synergy_loewe'].astype(float)
        # CHECK DRUG COMBINATION SCORE IN DIFFERENT MATRICS
        oneil_comb_num_df = oneil_comb_df[['ic50_row', 'ic50_col', 'ri_row', 'ri_col', 'css_row', 'css_col', 'css_ri', 'S_sum', 'S_mean','S_max', 'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss']]
        oneil_comb_num_df.apply(minMax)
        # AVERAGE THE SCORE IN [synergy_loewe]
        oneil_comb_syn_df = oneil_comb_df[['block_id', 'drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe', 'study_name']]
        oneil_comb_syn_df = oneil_comb_syn_df.sort_values(by = ['block_id'])
        oneil_comb_syn_new_df = oneil_comb_syn_df.groupby(['drug_row', 'drug_col', 'cell_line_name']).agg({'synergy_loewe':'mean'}).reset_index()
        oneil_comb_syn_new_df.apply(minMax)
        # ax = oneil_comb_syn_new_df['synergy_loewe'].plot.kde()
        oneil_comb_syn_new_df = oneil_comb_syn_new_df.rename(columns={'drug_row': 'Drug A', 'drug_col': 'Drug B', 
                                                'cell_line_name': 'Cell Line Name', 'synergy_loewe': 'Score'})
        ### DROP ['UWB1289+BRCA1'] CELL LINE, REDUCING NUMBER OF ROWS FROM [22737] TO [22154] 
        oneil_comb_syn_new_df.drop(oneil_comb_syn_new_df[oneil_comb_syn_new_df['Cell Line Name'] == 'UWB1289+BRCA1'].index, inplace = True)
        oneil_comb_syn_new_df.drop(oneil_comb_syn_new_df[oneil_comb_syn_new_df['Cell Line Name'] == 'MSTO'].index, inplace = True)
        ### PROFILE [Number of Drugs / Number of Cell Lines]
        drug_list = list(set(list(oneil_comb_syn_new_df['Drug A']) + list(oneil_comb_syn_new_df['Drug B'])))
        cell_line_list = list(set(list(oneil_comb_syn_new_df['Cell Line Name'])))
        print('----- NUMBER OF DRUGS IN O\'NEIL DATASET: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF CELL LINES O\'NEIL DATASET: ' + str(len(cell_line_list)) + ' -----')
        print(oneil_comb_syn_new_df.shape)
        oneil_comb_syn_new_df.to_csv('../' + dataset +  '/init_data/oneil_dl_input.csv', index=False, header=True)

    def parse_all_drugcomb_fi(self):
        dataset = self.dataset
        ### INTIALIZE [O'NEIL DrugScreen Data]
        print('----- READING DrugComb_fi DRUG SCREEN RAW DATA -----')
        if os.path.exists('../' + dataset +  '/init_data') == False:
            os.mkdir('../' + dataset +  '/init_data')
        drugcomb_fi_df = pd.read_csv('../' + dataset +  '/raw_data/drugcomb-fi/summary_v_1_5.csv')
        ### SELECT ONLY [ONEIL] DATASET
        drugcomb_fi_df = drugcomb_fi_df.loc[drugcomb_fi_df['drug_col'].notnull()]
        drugcomb_fi_df['synergy_loewe'] = drugcomb_fi_df['synergy_loewe'].astype(float)
        # CHECK DRUG COMBINATION SCORE IN DIFFERENT MATRICS
        drugcomb_fi_num_df = drugcomb_fi_df[['ic50_row', 'ic50_col', 'ri_row', 'ri_col', 'css_row', 'css_col', 'css_ri', 'S_sum', 'S_mean','S_max', 'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss']]
        drugcomb_fi_num_df.apply(minMax)
        # AVERAGE THE SCORE IN [synergy_loewe]
        drugcomb_fi_syn_df = drugcomb_fi_num_df[['block_id', 'drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe', 'study_name']]
        drugcomb_fi_syn_df = drugcomb_fi_syn_df.sort_values(by = ['block_id'])
        drugcomb_fi_syn_new_df = drugcomb_fi_syn_df.groupby(['drug_row', 'drug_col', 'cell_line_name']).agg({'synergy_loewe':'mean'}).reset_index()
        drugcomb_fi_syn_new_df.apply(minMax)
        # ax = oneil_comb_syn_new_df['synergy_loewe'].plot.kde()
        drugcomb_fi_syn_new_df = drugcomb_fi_syn_new_df.rename(columns={'drug_row': 'Drug A', 'drug_col': 'Drug B', 
                                                'cell_line_name': 'Cell Line Name', 'synergy_loewe': 'Score'})
        ### DROP ['UWB1289+BRCA1'] CELL LINE, REDUCING NUMBER OF ROWS FROM [22737] TO [22154] 
        drugcomb_fi_syn_new_df.drop(drugcomb_fi_syn_new_df[drugcomb_fi_syn_new_df['Cell Line Name'] == 'UWB1289+BRCA1'].index, inplace = True)
        drugcomb_fi_syn_new_df.drop(drugcomb_fi_syn_new_df[drugcomb_fi_syn_new_df['Cell Line Name'] == 'MSTO'].index, inplace = True)
        ### PROFILE [Number of Drugs / Number of Cell Lines]
        drug_list = list(set(list(drugcomb_fi_syn_new_df['Drug A']) + list(drugcomb_fi_syn_new_df['Drug B'])))
        cell_line_list = list(set(list(drugcomb_fi_syn_new_df['Cell Line Name'])))
        print('----- NUMBER OF DRUGS IN O\'NEIL DATASET: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF CELL LINES O\'NEIL DATASET: ' + str(len(cell_line_list)) + ' -----')
        print(drugcomb_fi_syn_new_df.shape)
        drugcomb_fi_syn_new_df.to_csv('../' + dataset +  '/init_data/oneil_dl_input.csv', index=False, header=True)

    def gdsc_rnaseq(self):
        dataset = self.dataset
        ### INTIALIZE [GDSC RNA Sequence Data]
        print('----- READING GDSC RNA Sequence RAW DATA -----')
        rna_df = pd.read_csv('../' + dataset +  '/raw_data/GDSC/rnaseq_20191101/rnaseq_fpkm_20191101.csv', low_memory=False)
        rna_df = rna_df.fillna('missing')
        rna_df.to_csv('../' + dataset +  '/init_data/gdsc_rnaseq.csv', index=False, header=True)
        print(rna_df.shape)
        # AFTER THIS NEED SOME MANUAL OPERATIONS TO CHANGE COLUMNS AND ROWS NAMES

    def gdsc_cnv(self):
        dataset = self.dataset
        cnv_df = pd.read_csv('../' + dataset +  '/raw_data/GDSC/cnv_20191101/cnv_gistic_20191101.csv', low_memory=False)
        cnv_df = cnv_df.fillna('missing')
        cnv_df.to_csv('../' + dataset +  '/init_data/gdsc_cnv.csv', index = False, header = True)
        print(cnv_df.shape)
        # AFTER THIS NEED SOME MANUAL OPERATIONS TO CHANGE COLUMNS AND ROWS NAMES

    def kegg(self):
        dataset = self.dataset
        kegg_pathway_df = pd.read_csv('../' + dataset +  '/raw_data/KEGG/full_kegg_pathway_list.csv')
        kegg_pathway_df = kegg_pathway_df[['source', 'target', 'pathway_name']]
        kegg_df = kegg_pathway_df[kegg_pathway_df['pathway_name'].str.contains('signaling pathway|signaling pathways', case=False)]
        print(kegg_df['pathway_name'].value_counts())
        print('----- NUMBER OF SIGNALING PATHWAYS IN KEGG: ' + str(len(kegg_df['pathway_name'].value_counts())) + ' -----')
        # import pdb; pdb.set_trace()
        kegg_df = kegg_df.rename(columns={'source': 'src', 'target': 'dest'})
        src_list = list(kegg_df['src'])
        dest_list = list(kegg_df['dest'])
        path_list = list(kegg_df['pathway_name'])
        # ADJUST ALL GENES TO UPPERCASE
        up_src_list = []
        for src in src_list:
            up_src = src.upper()
            up_src_list.append(up_src)
        up_dest_list = []
        for dest in dest_list:
            up_dest = dest.upper()
            up_dest_list.append(up_dest)
        up_kegg_conn_dict = {'src': up_src_list, 'dest': up_dest_list}
        up_kegg_df = pd.DataFrame(up_kegg_conn_dict)
        up_kegg_df = up_kegg_df.drop_duplicates()
        up_kegg_df.to_csv('../' + dataset +  '/init_data/up_kegg.csv', index=False, header=True)
        kegg_gene_list = list(set(list(up_kegg_df['src']) + list(up_kegg_df['dest'])))
        print('----- NUMBER OF GENES IN KEGG: ' + str(len(kegg_gene_list)) + ' -----')
        print(up_kegg_df.shape)

        up_kegg_path_conn_dict = {'src': up_src_list, 'dest': up_dest_list, 'path': path_list}
        up_kegg_path_df = pd.DataFrame(up_kegg_path_conn_dict)
        up_kegg_path_df = up_kegg_path_df.drop_duplicates()
        up_kegg_path_df.to_csv('../' + dataset +  '/init_data/up_kegg_path.csv', index=False, header=True)
        kegg_gene_list = list(set(list(up_kegg_path_df['src']) + list(up_kegg_path_df['dest'])))
        print('----- NUMBER OF GENES IN KEGG: ' + str(len(kegg_gene_list)) + ' -----')
        print(up_kegg_path_df.shape)

    def drugbank(self):
        dataset = self.dataset
        # INITIALIZE THE DRUG BANK INTO [.csv] FILE
        drugbank_df = pd.read_table('../' + dataset +  '/raw_data/DrugBank/drug_tar_drugBank_all.txt', delimiter='\t')
        drug_list = list(set(list(drugbank_df['Drug'])))
        target_gene_list = list(set(list(drugbank_df['Target'])))
        print('----- NUMBER OF DRUGS IN DrugBank: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF GENES IN DrugBank: ' + str(len(target_gene_list)) + ' -----')
        drugbank_df.to_csv('../' + dataset +  '/init_data/drugbank.csv', index=False, header=True)


##### Dataset selection
dataset = 'data-drugcomb-fi'
# dataset = 'data-DrugCombDB'
# dataset = 'data-nci'
# dataset = 'data-oneil'

# ReadFile().nci_combo_input()
ReadFile(dataset).parse_drugcomb_fi()
# ReadFile(dataset).gdsc_rnaseq()
# ReadFile(dataset).gdsc_cnv()

ReadFile(dataset).kegg()
# ReadFile(dataset).drugbank()