import numpy as np
standardized_vecs = np.zeros((n, 2048)) # (n, 2048) matrix of standardized latent vectors
pca_vecs = np.zeros((n,3)) # (n,3) matrix of PCA space
#assume that these share an index, i.e., standardized[k] corresponds to pca[k]
input = np.zeros((3,)) # (3,) point on PCA space

k = 4

distances = np.linalg.norm(pca_vecs - input,axis=1)
weights = 1/(distances+1e-3)            #regularize to avoid division by 0
min_entries = np.argpartition(distances,k-1)[:k]
min_std, min_pca, min_wts = standardized_vecs[min_entries], pca_vecs[min_entries], weights[min_entries]

reg_wts = (min_wts / min_wts.sum(axis=0)).reshape(k, 1)# make weights sum to 1
synth_pt = np.sum(reg_wts * min_std,axis=0)

np.save("user_pt.npy",synth_pt)
