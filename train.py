from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn as nn
import math, os
import torch
from utils.train_utils import *
from utils.train_cascade import local_train

#########NOVA######### PUSH TEST

parser = ArgumentParser()
# hyper-params of dataset
parser.add_argument('--log_dir', type=str, default='./try_run', help='Folder in which to save model and logs')
parser.add_argument('--root', type=str, default="./data/")
parser.add_argument('--data_type', type=str, default="qm9")
# Training arguments
parser.add_argument('--train_global', action='store_true', default=False, help='If set, train global refiner; else, train local')
parser.add_argument('--sigma_min', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--sigma_max', type=float, default=1, help='Initial learning rate')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for preprocessing')
parser.add_argument('--optimizer', type=str, default='adam', help='Adam optimiser only one supported')
parser.add_argument('--scheduler', type=str, default='plateau', help='LR scehduler: plateau or none')
parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience of plateau scheduler')
parser.add_argument('--limit_train_mols', type=int, default=0, help='Limit to the number of molecules in dataset, 0 uses them all')
# hyper-params of model
parser.add_argument('--model_name', type=str, default="default")
# Feature arguments
parser.add_argument('--in_node_features', type=int, default=53, help='Dimension of node features: 77 for drugs and xl, 47 for qm9')
parser.add_argument('--in_edge_features', type=int, default=8, help='Dimension of edge feature (do not change)')
parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Dimension of sinusoidal embedding of sigma')
parser.add_argument('--radius_embed_dim', type=int, default=50, help='Dimension of embedding of distances')
# Model arguments
parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of interaction layers')
parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
parser.add_argument('--ns', type=int, default=32, help='Number of hidden features per node of order 0')
parser.add_argument('--nv', type=int, default=8, help='Number of hidden features per node of orser >0')
parser.add_argument('--no_residual', action='store_true', default=False, help='If set, it removes residual connection')
parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
args = parser.parse_args()

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

print(args)
local_train(args=args)

