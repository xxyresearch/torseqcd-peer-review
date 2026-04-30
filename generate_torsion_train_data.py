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

from utils.inference_utils import generate_conformer_mols
from utils.dataloader_utils import collate_batch_local_test
from utils.assemble import assemble_frag, set_conformer_positions
from diffusion.ddpm_inference import local_inference
from diffusion.score_model import TensorProductScoreModel_local

parser = ArgumentParser()
parser.add_argument('--ckpt', type=str, default='try_run/best_model_local.pt', help='ckpt_position')
parser.add_argument('--root', type=str, default="./data/")
parser.add_argument('--data_type', type=str, default="qm9")
parser.add_argument('--mode', type=str, default="train")
# Training arguments
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--device', type=str, default="cuda:0")
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
                    mol          = data.mol,
                    x            = data.x,
                    edge_index   = data.edge_index,
                    edge_attr    = data.edge_attr,
                    sg_node_idx  = data.sub_node_idx,
                    sg_edge_idx  = data.sub_edge_idx,
                    sg_edge_attr = torch.cat(data.sub_edge_attr, dim=0),
                    sg_node_list = data.sub_node_idx_list,
                    shared_pair  = data.shared_pair,
                    pos          = data.pos
                    )

    SG_POS = []
    for node_idx in processed.sg_node_list:
        sg_pos = data.pos[node_idx]
        sg_pos_center = sg_pos - sg_pos.mean(dim=0, keepdim=True)
        SG_POS.append(sg_pos_center)
    
    processed.z_sg_pos = torch.cat(SG_POS, dim=0)
    return processed

model = TensorProductScoreModel_local(in_node_features=args.in_node_features, in_edge_features=args.in_edge_features,
                            ns=args.ns, nv=args.nv, sigma_embed_dim=args.sigma_embed_dim,
                            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                            num_conv_layers=args.num_conv_layers,
                            max_radius=args.max_radius, radius_embed_dim=args.radius_embed_dim,
                            scale_by_sigma=args.scale_by_sigma,
                            use_second_order_repr=args.use_second_order_repr,
                            residual=not args.no_residual, batch_norm=not args.no_batch_norm)
ckpt_path = args.ckpt
ckpt = torch.load(ckpt_path, map_location=args.device)
state_dict = ckpt.get("model", ckpt) 
model.load_state_dict(state_dict)
inference_method = local_inference(model, T=1000, start_t = 100, device=args.device)
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

for fn in tqdm(file_names):
    save_path = f"data/QM9/generated_local/generated_conformer_{args.mode}_{pickle_id}.pkl"
    
    inference_data = []
    with open(osp.join(data_dir, fn), 'rb') as f:
        raw = pickle.load(f)
    if len(raw) >= 30: raw = raw[:30]
    smiles = raw[0].smiles

    cleaned_conformers = [data.mol for data in raw ]
    generate_mols = generate_conformer_mols(cleaned_conformers, num_confs=1, seed=42)
    pos_before = raw[0].pos

    if generate_mols  == []:
        print(f'{smiles} cannot generate rd conformers')
        continue
    
    for i, gen_mol in enumerate(generate_mols):
        data = raw[i]
        data.mol = gen_mol
        data.pos = torch.tensor(gen_mol.GetConformer().GetPositions(), dtype=torch.float32)
        inference_data.append(process_data(data))

    inference_loader = DataLoader(dataset = inference_data, batch_size= args.batch_size, shuffle = False, num_workers=args.num_workers, collate_fn=collate_batch_local_test)
    for batch_idx, loader in enumerate((inference_loader)):
        try:
            loader = inference_method.sample_ddim(loader, ddim_steps=10) 
        except:
            print(f'{smiles} generation failure')
    
        for conf_id, mol in enumerate(loader.mol_list):
            smiles = loader.smiles_list[conf_id]
            mask = (loader.batch_sg == conf_id)
            pos = loader.z_sg_pos[mask]
            sg_node_list = loader.sg_node_list[conf_id]
            if len(sg_node_list) > 1:
                pos = assemble_frag(sg_node_list, pos, mol)
            mol_new = deepcopy(mol)
            mol_new = set_conformer_positions(mol_new, pos)
            if smiles not in output_dict: output_dict[smiles] = []
            output_dict[smiles].append(mol_new) 

    smiles_dict[smiles] = save_path
    if len(output_dict.keys()) == 1000:
        os.makedirs("data/QM9/generated_local", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(output_dict, f)
        pickle_id += 1
        output_dict = {}

os.makedirs("data/QM9/generated_local", exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(output_dict, f)
pickle_id += 1
output_dict = {}

with open(f'smiles_dict_{args.mode}.json', 'w') as f:
    json.dump(smiles_dict, f, indent=4)


