from utils import *
from trainer import *
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

seed=2024
fix_seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Run spatial omics analysis pipeline')
    
    parser.add_argument('--file_fold', type=str, required=True,
                       help='Path to the data folder')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to save the output file')
    parser.add_argument('--n_cluster', type=str, required=True,
                       help='cluster number')
    parser.add_argument('--radius', type=str, default=0.2,
                       help='radius')
    parser.add_argument('--n_top_genes', type=int, default=3000,
                       help='Number of top highly variable genes to select (default: 3000)')
    parser.add_argument('--spatial_neighbors', type=int, default=3,
                       help='Number of spatial neighbors (default: 3)')
    parser.add_argument('--omics1_neighbors', type=int, default=10,
                       help='Number of neighbors for omics1 (default: 10)')
    parser.add_argument('--omics2_neighbors', type=int, default=10,
                       help='Number of neighbors for omics2 (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay (default: 0.1)')
    parser.add_argument('--epoch', type=int, default=5000,
                       help='Number of epochs (default: 5000)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training (default: cuda:0)')
    
    args = parser.parse_args()
    
    adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
    adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_ADT.h5ad')
    
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=args.n_top_genes)
    
    adata_omics2 = clr_normalize_each_cell(adata_omics2)
    sc.pp.scale(adata_omics2)
    
    adata_omics1, adata_omics2 = construct_neighbor_graph(
        adata_omics1, adata_omics2,
        omics1_comps=adata_omics2.shape[1]-1,
        omics2_comps=adata_omics2.shape[1]-1,
        spatial_neighbors=args.spatial_neighbors,
        omics1_neighbors=args.omics1_neighbors,
        omics2_neighbors=args.omics2_neighbors,
        radius=args.radius,
    )
    
    train(
        adata_omics1, adata_omics2,
        save_path=args.save_path,
        n_cluster=args.n_cluster,
        lr=args.lr,
        weight_decay=args.weight_decay,
        Epoch=args.epoch,
        device=args.device
    )

if __name__ == '__main__':
    main()
