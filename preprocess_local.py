from argparse import ArgumentParser
import glob
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch_geometric.data import Data
from copy import deepcopy
import torch


from utils.preprocess_utils import open_pickle, get_full_smiles, clean_data
from utils.preprocess_utils import featurize_mol, get_subgraph_attr, add_chiral_edge_order_feature
from utils.preprocess_utils import save_pyg_data_to_pkl

parser = ArgumentParser()
parser.add_argument('--conf_num', type=int, default=30)
parser.add_argument('--root', type=str, default="./")
parser.add_argument('--data_type', type=str, default="qm9")
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--para_seed_num', type=int, default=0, help='Parallel computing preprocessing seeds')
parser.add_argument('--para_batch_num', type=int, default=1, help='Parallel computing preprocessing batch number')
args = parser.parse_args()

if args.data_type == 'qm9':
    atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    data_dir = args.root + 'data/QM9/qm9/'
    std_pkl_dir = args.root + 'data/QM9/standardized_pickles/'
    split_path = args.root + 'data/QM9/split.npy'
else:
    atom_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
        'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
        'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
        'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
    data_dir = args.root  + 'data/DRUGS/drugs/'
    std_pkl_dir = args.root + 'data/DRUGS/standardized_pickles'
    split_path = args.root + 'data/DRUGS/split.npy'

all_smiles = sorted(glob.glob(osp.join(data_dir, '*.pickle')))
all_std_files = sorted(glob.glob(osp.join(std_pkl_dir , '*.pickle')))
split_idx = 0 if args.mode == 'train' else 1 if args.mode == 'val' else 2
split = np.load(split_path, allow_pickle=True)[split_idx]
pickle_files = [f for i, f in enumerate(all_smiles) if i in split]

for idx, pkl_file in enumerate(tqdm(pickle_files)):
    if idx % args.para_batch_num == args.para_seed_num:
        raw_data = open_pickle(pkl_file)
        smiles = raw_data['smiles']
        confs = raw_data['conformers']
        ref_conf =  confs[0]['rd_mol']
        full_smiles = get_full_smiles(ref_conf)
        cleaned_conformers, full_smiles = clean_data(ref_conf, confs)
        if cleaned_conformers == []:
            continue
        data = Data(smiles = smiles, mol = cleaned_conformers[0], conf_list = cleaned_conformers)
        data = featurize_mol(data, atom_types)
        data = add_chiral_edge_order_feature(data, data.mol)
        data = get_subgraph_attr(data)   # __-NO R/S 
        pos = []
        for conformer in cleaned_conformers:
            pos.append(torch.tensor(conformer.GetConformer().GetPositions(), dtype=torch.float32))
        data.pos = pos
        sg_pos_batch = []
        for idx_sub, sub_node_idx in enumerate(data.sub_node_idx_list):
            sg_pos_batch.append(torch.ones_like(sub_node_idx)*idx_sub)
        data.sg_pos_batch = torch.concat(sg_pos_batch, dim=0)
        save_pyg_data_to_pkl(data = data, idx = idx, args=args, task='local')