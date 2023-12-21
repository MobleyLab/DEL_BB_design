import argparse
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def run_2D_dist(ref, test, output):
    ref_df = pd.read_csv(ref)
    test_df = pd.read_csv(test)

    ref_df['mol'] = ref_df['truncate_SMILES'].apply(Chem.MolFromSmiles)
    ref_df['fp'] = ref_df['mol'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048))

    test_df['mol'] = test_df['truncate_SMILES'].apply(Chem.MolFromSmiles)
    test_df['fp'] = test_df['mol'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048))

    distance_matrix = np.zeros((len(ref_df), len(test_df)), dtype=np.float32)
    for i, ref in enumerate(list(ref_df['fp'])):
        for j, test in enumerate(list(test_df['fp'])):
            distance_matrix[i,j] = 1 - DataStructs.TanimotoSimilarity(ref,test)

    np.save(output, distance_matrix)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--ref',
            action='store',
            type=str,
            help='path to csv file containing SMILES of reference compounds',
            required=True)

    my_parser.add_argument('--test',
            action='store',
            type=str,
            help='path to csv file containing SMILES of test compounds',
            required=True)

    my_parser.add_argument('--output',
            action='store',
            type=str,
            help='path to output distance matrix',
            required=True)

    args = my_parser.parse_args()
    run_2D_dist(args.ref, args.test, args.output)
