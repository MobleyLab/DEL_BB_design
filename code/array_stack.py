import numpy as np

new = np.load('files/dist_mat_0.npy')

for i in range(1, 50):
    x = np.load(f'files/dist_mat_{i}.npy')
    new = np.hstack([new, x])

np.save('files/truncates_dist_mat.npy', new) 

