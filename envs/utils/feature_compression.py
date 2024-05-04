import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def pca_compression(reps, compress_dim=32):
    pca = PCA(n_components=compress_dim)
    reps = torch.tensor(pca.fit_transform(reps), dtype=torch.float32)
    return reps
