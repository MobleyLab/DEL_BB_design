import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, PandasTools, SaltRemover, Fingerprints, Scaffolds
from rdkit.Chem.Fingerprints import FingerprintMols

# Calculate fingerprints for all compounds
df = pd.read_csv("enamine_pamine_smiles.csv")
df['RDMol'] = df['SMILES'].apply(Chem.MolFromSmiles)
df['fp'] = df['RDMol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048))
fps = list(df['fp'])

sim = []
# compare all fp pairwise without duplicates
for n in range(len(df)-1): # -1 so the last fp will not be used
    s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n+1:]) # +1 compare with the next to the last fp
    # collect the SMILES and values
    for m in range(len(s)):
        sim.append(s[m])

# Save to numpy array
sim_array = np.zeros((len(df),len(df)), dtype=np.half)
a,b = np.triu_indices(len(df), k=1)
sim_array[a,b] = sim
sim_array[b,a] = sim

np.save('pamine_sim_2D.npy', sim_array)
