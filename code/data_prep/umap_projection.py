import numpy as np
import pandas as pd
import umap

# Load in data
df = pd.read_csv('../../files/all_truncates.csv')

# Truncate distance matrix generated from running `run_2D_dist.py` script
dist_mat = np.load('../../files/truncates_dist_mat.npy')

U = umap.UMAP(random_state=0, metric='precomputed')
transform = U.fit(dist_mat)
df['X'] = transform.embedding_[:,0]
df['Y'] = transform.embedding_[:,1]

df.to_csv('../../files/truncates_UMAP.csv', index=False)
