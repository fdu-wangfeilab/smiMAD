import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from layers import *
from utils import *
from tqdm import tqdm
from scipy.sparse import csr_matrix

def train(adata_omics1,adata_omics2,save_path='',n_cluster=5,Epoch = 5000,lr = 0.001,weight_decay = 0.01,device='cuda:0',other_type='adt'):

    if isinstance(adata_omics1.X, csc_matrix) or isinstance(adata_omics1.X, csr_matrix):
        RNA=torch.tensor(adata_omics1.X.toarray()).float().to(device)
    else:
        RNA=torch.tensor(adata_omics1.X).float().to(device)

    if other_type=='adt':
        if isinstance(adata_omics2.X, csc_matrix) or isinstance(adata_omics2.X, csr_matrix):
            ADT=torch.tensor(adata_omics2.X.toarray()).float().to(device)
        else:
            ADT=torch.tensor(adata_omics2.X).float().to(device)
    else:
        if isinstance(adata_omics2.X, csc_matrix) or isinstance(adata_omics2.X, csr_matrix):
            ATAC = adata_omics2.X.toarray()
            ATAC[ATAC != 0] = 1
            ATAC=torch.tensor(ATAC).to(device)
        else:
            ATAC = adata_omics2.X
            ATAC[ATAC != 0] = 1
            ATAC=torch.tensor(ATAC).to(device)
    
    spatial_adj=torch.tensor(adata_omics1.obsm['adj_spatial']).float().to(device)
    adj_feature_f1=torch.tensor(adata_omics1.obsm['adj_feature']).float().to(device)
    adj_feature_f2=torch.tensor(adata_omics2.obsm['adj_feature']).float().to(device)
    adjacency=torch.tensor(adata_omics2.obsm['adjacency']).float().to(device)

    if other_type=='adt':
        model=MMS(RNA.shape[1],ADT.shape[1]).to(device)
    else:
        model=MMS(RNA.shape[1],ATAC.shape[1],modality_type='atac').to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    
    tloss = 0
    progress_bar = tqdm(range(Epoch), desc="Training Progress, loss: 0.0000")
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()

        if other_type=='adt':
            pi, dispersion, mean, modality2_reconstruction,spatial_specific1,spatial_shared1,feature_shared1,feature_specific1,spatial_specific2,spatial_shared2,feature_shared2,feature_specific2,fused_latent,_,_,_=model(RNA,ADT,spatial_adj,adj_feature_f1,adj_feature_f2)
        else:
            pi, dispersion, mean, modality2_reconstruction,spatial_specific1,spatial_shared1,feature_shared1,feature_specific1,spatial_specific2,spatial_shared2,feature_shared2,feature_specific2,fused_latent,_,_,_=model(RNA,ATAC,spatial_adj,adj_feature_f1,adj_feature_f2)
    
        rec_1 = rna_loss(y_true=RNA, y_pred=mean, pi=pi, theta=dispersion, ridge_lambda=0.0, scale_factor=1.0, mean=True)
        
        if other_type=='adt':
            rec_2 = adt_loss(modality2_reconstruction, ADT)
        else:
            rec_2 = atac_loss(modality2_reconstruction, ATAC)

        
        mmd1 = mmd_gaussian_loss(spatial_shared1,feature_shared1)
        mmd2 = mmd_gaussian_loss(spatial_shared2,feature_shared2)
        spl = spatial_consistency_loss(fused_latent,adjacency, epoch)
        decl = decorrelation_loss(spatial_specific1,feature_specific1)+decorrelation_loss(spatial_specific2,feature_specific2)
        
        loss = rec_1 + rec_2 + mmd1 + mmd2 + spl + decl

        loss.backward()
        optimizer.step()
    
        progress_bar.set_description(f"Training Progress, loss: {loss:.4f},rec1: {rec_1:.4f},rec2: {rec_2:.4f}, mmd1: {mmd1:.4f},mmd2: {mmd2:.4f}, sp: {spl:.4f}, dec: {decl:.4f}")

    if other_type=='adt': 
        with torch.no_grad():
            _, _, _, _, _, _, _, _,_,_,_,_,fused_latent,attention_weights1,attention_weights2,attention_weights_fusion=model(RNA,ADT,spatial_adj,adj_feature_f1,adj_feature_f2)

    else:
        with torch.no_grad():
            _, _, _, _, _, _, _, _,_,_,_,_,fused_latent,attention_weights1,attention_weights2,attention_weights_fusion=model(RNA,ATAC,spatial_adj,adj_feature_f1,adj_feature_f2)
    
    adata_omics2.obsm['fused_latent'] = fused_latent.detach().cpu().numpy()
    adata_omics2.obsm['attention_weights1'] = attention_weights1.detach().cpu().numpy()
    adata_omics2.obsm['attention_weights2'] = attention_weights2.detach().cpu().numpy()
    adata_omics2.obsm['attention_weights_fusion'] = attention_weights_fusion.detach().cpu().numpy()
    clustering(adata_omics2,n_cluster)
    
    adata_omics2.write_h5ad(save_path)
