from layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMS_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(MMS_Encoder, self).__init__()

        self.spatial_branch = MultiGCN(input_dim, hidden_dim, output_dim, dropout)
        self.feature_branch = MultiGCN(input_dim, hidden_dim, output_dim, dropout)
        self.shared_branch = MultiGCN(input_dim, hidden_dim, output_dim, dropout)

        self.gate_mechanism = self_gating(output_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout)
        )

        self.residual_scaling = nn.Parameter(torch.ones(1))  # Learnable residual weight

    def compute_embeddings(self, features, graph, specific_branch, shared_branch):
        return {
            "specific": specific_branch(features, graph),
            "shared": shared_branch(features, graph)
        }

    def encode(self, features, spatial_graph, feature_graph):
        spatial_embeddings = self.compute_embeddings(features, spatial_graph, self.spatial_branch, self.shared_branch)
        feature_embeddings = self.compute_embeddings(features, feature_graph, self.feature_branch, self.shared_branch)

        modality_embeddings = torch.stack(
            [
                spatial_embeddings["specific"],
                spatial_embeddings["shared"],
                feature_embeddings["shared"],
                feature_embeddings["specific"]
            ],
            dim=1
        )

        fused_representation, attention_weights = self.gate_mechanism(modality_embeddings)
        fused_output = self.residual_scaling * self.projection_head(fused_representation)
        
        return fused_output, spatial_embeddings["specific"], spatial_embeddings["shared"], feature_embeddings["shared"], feature_embeddings["specific"], attention_weights

    def forward(self, features, spatial_graph, feature_graph):
        return self.encode(features, spatial_graph, feature_graph)


class MMS(nn.Module):
    def __init__(self, omics1_dim, omics2_dim, hidden_dim=64, latent_dim=16, modality_type='adt'):
        super(MMS, self).__init__()

        self.omics1_encoder = MMS_Encoder(omics1_dim, hidden_dim, latent_dim)
        self.omics2_encoder = MMS_Encoder(omics2_dim, hidden_dim, latent_dim)

        self.latent_gating = self_gating(latent_dim)

        self.omics1_decoder = RNA_Decoder(latent_dim, hidden_dim, omics1_dim)

        if modality_type.lower() == 'adt':
            self.omics2_decoder = ADT_Decoder(latent_dim, hidden_dim, omics2_dim)
        else:
            self.omics2_decoder = ATAC_Decoder(latent_dim, hidden_dim, omics2_dim)

        self.modality_type=modality_type

    def forward(self, omics1_features, omics2_features, spatial_graph, feature_graph_omics1, feature_graph_omics2):

        latent_omics1, spatial_specific1, spatial_shared1, feature_shared1, feature_specific1, attention_weights1 = self.omics1_encoder(
            omics1_features, spatial_graph, feature_graph_omics1
        )
        latent_omics2, spatial_specific2, spatial_shared2, feature_shared2, feature_specific2, attention_weights2 = self.omics2_encoder(
            omics2_features, spatial_graph, feature_graph_omics2
        )

        fused_latent, attention_weights_fusion = self.latent_gating(
            torch.stack([latent_omics1, latent_omics2], dim=1)
        )

        pi, dispersion, mean = self.omics1_decoder(fused_latent)
        modality2_reconstruction = self.omics2_decoder(fused_latent)

        if self.modality_type=='atac':
            modality2_reconstruction = F.sigmoid(modality2_reconstruction)
        

        return (
            pi,
            dispersion,
            mean,
            modality2_reconstruction,
            spatial_specific1,
            spatial_shared1,
            feature_shared1,
            feature_specific1,
            spatial_specific2,
            spatial_shared2,
            feature_shared2,
            feature_specific2,
            fused_latent,
            attention_weights1,
            attention_weights2,
            attention_weights_fusion
        )


