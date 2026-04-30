import os
import os.path as osp
import json
import pickle
from copy import deepcopy
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from utils.dataloader_utils import collate_batch_global_test
from utils.assemble import set_conformer_positions
from utils.preprocess_utils import open_pickle
from diffusion.ddpm_inference import global_inference
from diffusion.score_model import TensorProductScoreModel_global

parser = ArgumentParser()
parser.add_argument('--ckpt', type=str, default='try_run/best_model_global.pt', help='ckpt_position')
parser.add_argument('--root', type=str, default="./data/")
parser.add_argument('--data_type', type=str, default="qm9")
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--ddpm', action='store_true', default=False, help='If set, use ddpm')
# Training arguments
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--start_t', type=int, default=100)
parser.add_argument('--ddim_step', type=int, default=10)
parser.add_argument('--device', type=str, default="cuda:1")
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for preprocessing')
parser.add_argument('--sigma_min', type=float, default=0.01*3.14, help='Minimum sigma used for training')
parser.add_argument('--sigma_max', type=float, default=3.14, help='Maximum sigma used for training')
parser.add_argument('--limit_train_mols', type=int, default=0, help='Limit to the number of molecules in dataset, 0 uses them all')
# hyper-params of model
parser.add_argument('--model_name', type=str, default="default")
# Feature arguments
parser.add_argument('--in_node_features', type=int, default=48, help='Dimension of node features: 74 for drugs and xl, 44 for qm9')
parser.add_argument('--in_edge_features', type=int, default=8, help='Dimension of edge feature (do not change)')
parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Dimension of sinusoidal embedding of sigma')
parser.add_argument('--radius_embed_dim', type=int, default=50, help='Dimension of embedding of distances')
# Model arguments
parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of interaction layers')
parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
parser.add_argument('--ns', type=int, default=32, help='Number of hidden features per node of order 0')
parser.add_argument('--nv', type=int, default=8, help='Number of hidden features per node of orser >0')
parser.add_argument('--no_residual', action='store_true', default=False, help='If set, it removes residual connection')
parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
args = parser.parse_args()

def process_data(data):
    processed = Data(
                    smiles       = data.smiles,
                    mol          = data.c2_mol,
                    x            = data.x,
                    edge_index   = data.edge_index,
                    edge_attr    = data.edge_attr,
                    pos          = torch.tensor(data.c2_pos, dtype=torch.float32)
                    )
    return processed

model = TensorProductScoreModel_global(in_node_features=args.in_node_features, in_edge_features=args.in_edge_features,
                            ns=args.ns, nv=args.nv, sigma_embed_dim=args.sigma_embed_dim,
                            num_conv_layers=args.num_conv_layers,
                            max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim,
                            use_second_order_repr=args.use_second_order_repr,
                            residual=not args.no_residual, batch_norm=not args.no_batch_norm)
ckpt_path = args.ckpt
ckpt = torch.load(ckpt_path, map_location=args.device)
state_dict = ckpt.get("model", ckpt) 
model.load_state_dict(state_dict)
inference_method = global_inference(model, T=1000, start_t = args.start_t, device=args.device)
inference_method.model.eval()

if args.data_type == 'qm9':
    data_dir = f"{args.root}qm9_{args.mode}_local/"
else:
    data_dir = f"{args.root}Drugs_{args.mode}_local/"

data_dir  = data_dir
file_names = sorted(os.listdir(data_dir))

output_dict = {}
smiles_dict = {}
pickle_id = 0
generated_c2 = open_pickle('data/QM9/test_cascade2_data.pkl')

for smiles in tqdm(generated_c2.keys()):

    raw_data = generated_c2[smiles]
    processed_data = []
    for i in raw_data:
        processed_data.append(process_data(i))
    #print(processed_data)
    inference_loader = DataLoader(dataset = processed_data, batch_size= args.batch_size, shuffle = False, num_workers=args.num_workers, collate_fn=collate_batch_global_test)
    for batch_idx, loader in enumerate(tqdm(inference_loader)):
        try:
            if args.ddpm:
                loader = inference_method.sample_ddpm(loader) 
            else:
                loader = inference_method.sample_ddim(loader, ddim_steps=args.ddim_step) 
        except:
            print(f'{smiles} generation failure')
    
        for conf_id, mol in enumerate(loader.mol_list):
            mask = (loader.batch == conf_id)
            pos = loader.pos[mask]
            mol_new = deepcopy(mol)
            mol_new = set_conformer_positions(mol_new, pos)
            if smiles not in output_dict: output_dict[smiles] = []
            output_dict[smiles].append(mol_new) 

    # if len(output_dict.keys()) >= 10:
    #     break

os.makedirs("data/QM9", exist_ok=True)

with open("data/QM9/test_cascade3_output.pkl", "wb") as f:
    pickle.dump(output_dict, f)

