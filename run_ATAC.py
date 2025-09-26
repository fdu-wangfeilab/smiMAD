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
    parser.add_argument('--datatype', type=str, default='Simulation',
                       help='datatype')
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
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device to use for training (default: cuda:1)')
    
    args = parser.parse_args()
    
    adata_omics1,adata_omics2=pre_process(args.file_fold + 'adata_RNA.h5ad',
                                          args.file_fold + 'adata_peaks_normalized.h5ad',
                                          data_type=args.datatype,
                                      spatial_neighbors=args.spatial_neighbors,
                                      omics1_neighbors=args.omics1_neighbors,
                                      omics2_neighbors=args.omics2_neighbors,
                                      radius=args.radius
                                     )
    
    train(
        adata_omics1, adata_omics2,
        save_path=args.save_path,
        n_cluster=args.n_cluster,
        lr=args.lr,
        weight_decay=args.weight_decay,
        Epoch=args.epoch,
        device=args.device,
        other_type='atac'
    )


if __name__ == '__main__':
    main()