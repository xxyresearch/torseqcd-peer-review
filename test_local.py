from utils.preprocess_utils import open_pickle
from utils.conformer_match import get_torsion_angles, optimize_rotatable_bonds
from tqdm import tqdm
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx


groundtruth = open_pickle("data/QM9/test_mols.pkl")
generated = open_pickle("data/QM9/test_generate_mols.pkl")

Total_RMSD_median = []
Total_RMSD_avg = []
N = 2

for idx, smi in enumerate(tqdm(groundtruth.keys())):
    real_mols = groundtruth[smi]
    try:
        gen_mols = generated[smi]
        if len(gen_mols) >= N:
            gen_mols = random.sample(gen_mols, N)

        RMSD = []
        for g_mol in gen_mols:
            new_rmsd = []
            for r_mol in real_mols :
                rotatable_bonds = get_torsion_angles(r_mol)
                if len(rotatable_bonds) != 0:
                    optimize_rotatable_bonds(r_mol, g_mol, rotatable_bonds, popsize=15, maxiter=15)  
                new_rmsd.append(AllChem.GetBestRMS(Chem.RemoveHs(g_mol), Chem.RemoveHs(r_mol)) )
            RMSD.append(np.min(np.array(new_rmsd)))
        Total_RMSD_avg.append(np.mean(RMSD))
        print(f'smi: {smi}, RMSD Lower bound: {np.mean(RMSD)}')
    except:
        print(smi)
        continue
    #print(Total_RMSD_median, Total_RMSD_avg)
    if idx%50 == 0: print(f'{idx}, {np.median(Total_RMSD_avg)}, {np.mean(Total_RMSD_avg)}')

print(f'Final Result, Median: {np.median(Total_RMSD_avg)}, Mean:{np.mean(Total_RMSD_avg)}')