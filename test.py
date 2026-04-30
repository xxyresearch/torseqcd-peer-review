import pickle
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os

parser = ArgumentParser()
parser.add_argument('--test_csv', type=str, default='./data/DRUGS/test_smiles.csv', help='Path to csv file with list of smiles')
parser.add_argument('--true_mols', type=str, default='./data/DRUGS/test_mols.pkl', help='Path to pickle file with ground truth conformers')
parser.add_argument('--n_workers', type=int, default=1, help='Numer of parallel workers')
parser.add_argument('--limit_mols', type=int, default=0, help='Limit number of molecules, 0 to evaluate them all')
#parser.add_argument('--dataset', type=str, default="drugs", help='Dataset: drugs, qm9 and xl')
parser.add_argument('--filter_mols', type=str, default=None, help='If set, is path to list of smiles to test')
parser.add_argument('--only_alignmol', action='store_true', default=False, help='If set instead of GetBestRMSD, it uses AlignMol (for large molecules)')
args = parser.parse_args([])

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def datalist_to_dict(data_list):
    smi_dict = {}
    for data in data_list:
        smi = data['smi']
        mol = data['mol']
        if smi not in smi_dict:
            smi_dict[smi] = []
        smi_dict[smi].append(mol)
    return smi_dict

# load ground truth
test_data = pd.read_csv(args.test_csv)  # this should include the corrected smiles
with open(args.true_mols, 'rb') as f:
    true_mols = pickle.load(f)
threshold = 0.75

# torsional diffusion
generate_confs_file_path= 'data/workdir/drugs_default/drugs_steps20.pkl'
generate_confs_dict_baseline  = load_pickle(generate_confs_file_path)
res_dict = {}
for smi in generate_confs_dict_baseline.keys():
    try:
        true_confs = true_mols[smi]
    except:
        print(f'{smi} cannot be found')
        continue
    
    gen_confs = generate_confs_dict_baseline[smi]
    res_value = []
    rmsd_array = []
    bad_cases = False
    for tc in tqdm(true_confs):
        rmsd_list = []
        for gc in gen_confs:
            try:
                rmsd_list.append(AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(gc)))
            except:
                bad_cases = True
        rmsd_array.append(rmsd_list)
    if bad_cases:
        print(f'{smi} meet error during get RMSD')
        continue

    res_value.append(np.mean(np.min(rmsd_array, axis=1)<=threshold))
    res_value.append( np.mean(np.min(rmsd_array, axis=1)))
    res_value.append(np.mean(np.min(rmsd_array, axis=0)<=threshold))
    res_value.append(np.mean(np.min(rmsd_array, axis=0)))
    res_dict[smi] = res_value

import os
folder_path = 'data/drugs_test_local_generated'
pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

# Load all .pkl files
data_list = []
for file_name in pkl_files:
    file_path = os.path.join(folder_path, file_name)
    data = load_pickle(file_path)
    data_list.append(data)


generate_confs_dict = datalist_to_dict(data_list)
ours_res = {}

for smi in res_dict.keys():
    try:
        true_confs = true_mols[smi]
    except:
        
        continue
    
    if smi in generate_confs_dict.keys():
        gen_confs = generate_confs_dict[smi]
    
    else:
        ours_res[smi] = res_dict[smi]

    res_value = []
    rmsd_array = []
    bad_cases = False
    for tc in tqdm(true_confs):
        rmsd_list = []
        for gc in gen_confs:
            try:
                rmsd_list.append(AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(gc)))
            except:
                bad_cases = True
        rmsd_array.append(rmsd_list)
    if bad_cases:
        ours_res[smi] = res_dict[smi]
        print(f'{smi} meet error during get RMSD, directly use tordiff and not change anything')
        continue

    res_value.append(np.mean(np.min(rmsd_array, axis=1)<=threshold))
    res_value.append( np.mean(np.min(rmsd_array, axis=1)))
    res_value.append(np.mean(np.min(rmsd_array, axis=0)<=threshold))
    res_value.append(np.mean(np.min(rmsd_array, axis=0)))
    ours_res[smi] = res_value

# Calculate mean and median for each column
# Convert dictionary values to a numpy array
res = np.array(list(ours_res.values()))  # This creates an N x 4 array

# Calculate mean and median for each column
mean_per_column = np.mean(res, axis=0)
median_per_column = np.median(res, axis=0)

# Format output
formatted_output = {}
columns = ['A', 'B', 'C', 'D']
for i, col in enumerate(columns):
    formatted_output[f'mean for {col}'] = mean_per_column[i]
    formatted_output[f'median for {col}'] = median_per_column[i]

# Print formatted output
for key, value in formatted_output.items():
    print(f"{key}: {value}")