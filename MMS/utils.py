import os,random,scipy,anndata,sklearn
import numpy as np
import torch
from torch.backends import cudnn
import torch.nn.functional as F
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from typing import Optional
from sklearn.decomposition import PCA
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph
import pandas as pd

def fix_seed(seed=2024):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  

def construct_neighbor_graph(adata_omics1, adata_omics2, adata_omics2_type='ADT', spatial_neighbors=3,
                             omics1_neighbors=20, omics2_neighbors=20, omics1_comps=10, omics2_comps=10, radius=0.2):
    # spatial
    spatial_coords = adata_omics1.obsm['spatial']
    spatial_adj = construct_adjacency_by_spatial(spatial_coords, n_neighbors=spatial_neighbors)
    adata_omics1.obsm['adj_spatial'] = spatial_adj
    adata_omics2.obsm['adj_spatial'] = spatial_adj

    #feature
    adata_omics1 = construct_adjacency_by_features(adata_omics1, n_neighbors=omics1_neighbors, n_comps=omics1_comps)
    adata_omics2 = construct_adjacency_by_features(adata_omics2, n_neighbors=omics2_neighbors,
                                                                       n_comps=omics2_comps,Mtype=adata_omics2_type)

    adata_omics2.obsm['adjacency'] = spatial_construct_graph(adata_omics1,radius=radius)
    return adata_omics1,adata_omics2

def construct_adjacency_by_spatial(spatial_coords, n_neighbors=3):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    
    n = spatial_coords.shape[0]
    rows = np.repeat(indices[:, 0], n_neighbors)
    cols = indices[:, 1:].flatten()
    
    adj = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n, n)).toarray()
    adj = adj + adj.T 
    return np.where(adj > 0, 1, 0)  


def spatial_construct_graph(
    adata,
    radius = 0.2,
    normalize = True,
):
    coords = adata.obsm['spatial']

    if normalize:
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = np.clip(max_vals - min_vals, a_min=np.finfo(float).eps, a_max=None)
        coords = (coords - min_vals) / range_vals

    sparse_graph = radius_neighbors_graph(
        X=coords,
        radius=radius,
        mode='connectivity',
        include_self=False,
        metric='euclidean'
    )

    adjacency = sparse_graph.toarray().astype(np.float32)

    return adjacency



def construct_adjacency_by_features(adata, n_neighbors=2,n_comps=10,Mtype='RNA'):
    if Mtype=='RNA' or Mtype=='ADT':
        pca = PCA(n_components=n_comps)
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            features = pca.fit_transform(adata.X.toarray()) 
        else:   
            features = pca.fit_transform(adata.X)
        
    else:
        features = adata.obsm['X_lsi'].copy()
        
    adj = kneighbors_graph(
        features, 
        n_neighbors=n_neighbors,
        mode="distance",
        metric="correlation",
        include_self=False
    ).toarray()
    
    adj = adj + adj.T

    adata.obsm['adj_feature'] = np.where(adj > 1, 1, adj)
    adata.obsm['feat'] = features
    return adata


def clustering(adata, num_cluster=5, used_obsm='fused_latent', modelNames='EEE', random_seed=2024):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res.astype('int')
    return adata

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""
    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     



def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:

    X = tfidf(adata.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:,1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   

def mmd_gaussian_loss(emb1, emb2, sigma=1.0):
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    def gaussian_kernel(x, y):
        x_sqnorms = torch.sum(x ** 2, dim=1, keepdim=True)
        y_sqnorms = torch.sum(y ** 2, dim=1, keepdim=True)
        pairwise_dist = x_sqnorms - 2 * torch.mm(x, y.t()) + y_sqnorms.t()
        return torch.exp(-pairwise_dist / (2 * sigma ** 2))
    k_11 = gaussian_kernel(emb1, emb1)
    k_22 = gaussian_kernel(emb2, emb2)
    k_12 = gaussian_kernel(emb1, emb2)
    return k_11.mean() + k_22.mean() - 2 * k_12.mean()


def spatial_consistency_loss(
    fused_latent, adjacency, epoch, total_epochs=5000,
    start_low=0.1, end_low=0.3,
    start_high=0.9, end_high=0.7,
):
    device = fused_latent.device
    eps: float = 1e-8

    similarity = torch.matmul(fused_latent, fused_latent.T)
    norm = torch.norm(fused_latent, p=2, dim=1).reshape((fused_latent.shape[0], 1))
    similarity = similarity / (norm @ norm.T + 1e-8)  # 防止除0
    similarity = similarity - torch.diag_embed(torch.diag(similarity))
 
    similarity = (similarity + 1) / 2  # 映射到 [0,1]

    # 动态 margin
    progress = epoch / total_epochs
    margin_low = start_low + (end_low - start_low) * progress
    margin_high = start_high + (end_high - start_high) * progress

    # 正样本
    pos_mask = adjacency > 0
    pos_violation = (similarity < margin_high) & pos_mask

    pos_loss = F.relu(margin_high - similarity) * pos_violation.float()
    pos_loss = pos_loss.sum() / (pos_violation.sum() + 1e-8)

    adjacency_neg = 1.0 - adjacency
    adjacency_neg.fill_diagonal_(0)
    neg_mask = adjacency_neg > 0
    neg_violation = (similarity > margin_low) & neg_mask

    neg_loss = F.relu(similarity - margin_low) * neg_violation.float()
    neg_loss = neg_loss.sum() / (neg_violation.sum() + 1e-8)

    return 0.5 * (pos_loss + neg_loss)

def decorrelation_loss(spec1, spec2):
    cos_sim = F.cosine_similarity(spec1, spec2, dim=1)
    return (cos_sim ** 2).mean()  


def rna_loss(y_true, y_pred, pi, theta, ridge_lambda=0.0, scale_factor=1.0, mean=True, eps=1e-10):
    theta = torch.minimum(theta, torch.tensor(1e6))
    y_pred = y_pred * scale_factor
    
    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - \
         torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + \
         y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
    nb_term = t1 + t2
    
    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
    nb_case = nb_term - torch.log(1.0 - pi + eps)
    
    result = torch.where(y_true < 1e-8, zero_case, nb_case)
    result += ridge_lambda * torch.square(pi)
    
    result = torch.nan_to_num(result, nan=float('inf'))
    
    return torch.mean(result) if mean else result
    
def atac_loss(p, target):
    return F.binary_cross_entropy(p, target)

def adt_loss(mean, target):
    return F.huber_loss(mean, target) 
