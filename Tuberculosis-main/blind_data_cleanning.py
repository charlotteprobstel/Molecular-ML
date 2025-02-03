import pandas as pd

# 258 compounds
Mtb_testset_2017 = pd.read_csv('/rds/general/user/qg622/home/Y2/Antibacterial/data/2017_Mtb_testset.csv')
# 1,196 compounds
Mtb_predictions_Vadim_100nM = pd.read_csv('/rds/general/user/qg622/home/Y2/Antibacterial/data/Mtb_predictions_Vadim_100nM.csv')
# 15,618 compounds
Mtb_published_regression_AC_Cleaned = pd.read_csv('/rds/general/user/qg622/home/Y2/Antibacterial/data/Mtb_published_regression_AC_Cleaned.csv')
# 1,193 compounds
RCB_Mtb_inhibition_20072019 = pd.read_csv('/rds/general/user/qg622/home/Y2/Antibacterial/data/RCB_Mtb_inhibition_2007-2019.csv')

combined_dataframe = pd.concat([
    Mtb_testset_2017, 
    Mtb_predictions_Vadim_100nM, 
    Mtb_published_regression_AC_Cleaned, 
    RCB_Mtb_inhibition_20072019
], ignore_index=True)

# Paths to the CSV files
pubchem_path = '/rds/general/user/qg622/home/Y2/Antibacterial/data/pubchem_blind_testset.csv'
chembl_path = '/rds/general/user/qg622/home/Y2/Antibacterial/data/chembl_blind_testset.csv'

# Reading the CSV files
pubchem_data = pd.read_csv(pubchem_path)
chembl_data = pd.read_csv(chembl_path)

train_smi = combined_dataframe['SMILES'].unique()

# Remove rows where 'canonical_smiles' in 'pubchem_data' are in 'train_smi'
pubchem_filtered = pubchem_data[~pubchem_data['canonical_smiles'].isin(train_smi)]
pubchem_filtered = pubchem_filtered[pubchem_filtered['standard_type'] != 'MIC']

# Remove rows where 'canonical_smiles' in 'chembl_data' are in 'train_smi'
chembl_filtered = chembl_data[~chembl_data['canonical_smiles'].isin(train_smi)]
