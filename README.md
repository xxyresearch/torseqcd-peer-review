# TorSeqCD
Implementation of TorSeqCD by Xiangyang Xu, Meng Liu, and Hongyang Gao 

If you have any question, please open issue or send us email at xyxu@iastate.edu

## Enviornment
Create a new conda enviornment and install pytorch and pyg; if there are any problems, you can check the package we used from package.txt. The GPU we use is 2080Ti.

    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install torch_geometric
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
    pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html

## Dataset and checkpoints
We use the same dataset and split as Torsional Diffusion (Jing et al), you can download dataset from [here](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7?usp=sharing).

or python download_data.py

## preprocess data
    python TorseqCD/preprocess_local.py

# Bond Angle and Length (Local) Training
    python train.py

## Prepare data and Torsional Angle (Global) Training
    python generate_torsion_train_data.py
    python train.py --train_global

## Sampling
    python inference_local.py
    python inference_global.py
    
## Testing
    python test.py 
    TorseqCD/preprocess_local_test.py

This is for paper "TorSeqCD: Torsion Sequential Modeling for Molecular 3D Conformation Generation" peer review.
If you have any problem, please contact xyxu@iastate.edu
