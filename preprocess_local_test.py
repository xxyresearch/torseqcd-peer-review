import pickle
import random
from multiprocessing import Pool
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data

from utils.preprocess_utils import (
    featurize_mol,
    add_chiral_edge_order_feature,
    get_subgraph_attr,
    save_pyg_data_to_pkl,
    open_pickle,
)
from utils.inference_utils import generate_conformer_mols

parser = ArgumentParser()
#parser.add_argument('--confs', type=str, required=True, help='Path to pickle file with generated conformers')
parser.add_argument('--test_csv', type=str, default='./data/QM9/test_smiles.csv', help='Path to csv file with list of smiles')
parser.add_argument('--true_mols', type=str, default='./data/QM9/test_mols.pkl', help='Path to pickle file with ground truth conformers')
parser.add_argument('--n_workers', type=int, default=1, help='Numer of parallel workers')
parser.add_argument('--limit_mols', type=int, default=0, help='Limit number of molecules, 0 to evaluate them all')
parser.add_argument('--dataset', type=str, default="qm9", help='Dataset: drugs, qm9 and xl')
parser.add_argument('--filter_mols', type=str, default=None, help='If set, is path to list of smiles to test')
parser.add_argument('--only_alignmol', action='store_true', default=False, help='If set instead of GetBestRMSD, it uses AlignMol (for large molecules)')
args = parser.parse_args()


def check_data(smiles, conformers):
    mol = Chem.MolFromSmiles(smiles)
    # basic molecule check
    if mol:
        canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    else:
        print(f'{smiles}, mols rdkit cannot intrinsically handle')
        return []
    if '.' in smiles:
        print(f'{smiles}, conformers with fragments')
        return []
    # skip mols with atoms with more than 4 neighbors for now
    n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
    if np.max(n_neighbors) > 4:
        return []
    
    # check reacted conformer
    cleaned_conformers = clean_confs(smi = canonical_smi, confs=conformers)
    if len(cleaned_conformers) == 0:
        print(f'{smiles}, has no non-reacted conf')
        return []
    # 
    return cleaned_conformers

def clean_confs(smi, confs):
    good_ids = []
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)

    return [confs[i] for i in good_ids]



def mols_to_chiral_smiles(mol_list):
    smiles_list = []
    for mol in mol_list:
        # Ensure stereochemistry is updated from 3D coordinates
        Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)
        
        # Generate SMILES with stereochemistry (isomericSmiles=True)
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        smiles_list.append(smi)
    return smiles_list

# load test data
test_set = open_pickle("data/QM9/test_mols.pkl")
args.data_type = 'qm9'
args.mode = 'test'
args.root = './'

test_data = pd.read_csv(args.test_csv)  # this should include the corrected smiles
with open(args.true_mols, 'rb') as f:
    true_mols = pickle.load(f)
# RMSD

atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

rdkit_smiles = test_data.smiles.values
corrected_smiles = test_data.corrected_smiles.values

mol_idx = 0
Rdkit_fail = 0
gt_error = 0

for smi, corrected_smi in tqdm(zip(rdkit_smiles, corrected_smiles)):
    conformers = true_mols[smi]
    cleaned_conformers = check_data(corrected_smi, conformers)
    if cleaned_conformers == []:
        gt_error += 1
        continue

    generate_mols = generate_conformer_mols(cleaned_conformers, seed=42)

    if generate_mols == []:
        Rdkit_fail += 1
        print('Rdkit fail to generate')
        continue

    if len(generate_mols) != 2* len(cleaned_conformers):
        print(f' smi {len(generate_mols)} !=  { 2* len(cleaned_conformers)}')
        

    # for conf_idx, conformer in enumerate(generate_mols):
    #     data = Data(smiles = smi, mol = conformer)
    #     data = featurize_mol(data, atom_types)
    #     data = add_chiral_edge_order_feature(data, data.mol)
    #     data, sub_node_idx_list = get_subgraph_attr(data)
    #     subpos = []
    #     pos = torch.tensor(conformer.GetConformer().GetPositions(), dtype=torch.float32)
    #     data.pos = pos
    #     # for sub_node_idx in sub_node_idx_list:
    #     #     sub_pos = pos[sub_node_idx]
    #     #     subpos.append(sub_pos)
    #     # data.sub_pos = subpos
    #     data.sub_node_idx_list = sub_node_idx_list
    #     #save_pyg_data_to_pkl(data = data, idx = f'{mol_idx}_{conf_idx}', args=args, task='local')
    
    mol_idx += 1
    
print(f'{mol_idx} mol generate successfully. {Rdkit_fail} mol rdkit fail generate. {gt_error} no good groundtruth')