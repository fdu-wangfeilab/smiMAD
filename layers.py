import torch,math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MultiGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.0, num_layers=2):
        super(MultiGCN, self).__init__()
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()  

        in_dim = input_dim
        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:  
                out_dim_layer = out_dim
            else:
                out_dim_layer = hidden_dim
                
            conv_layer = GraphConvolution(in_dim, out_dim_layer)
            self.conv_layers.append(conv_layer)
            
            in_dim = out_dim_layer

    def forward(self, x, adj):
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, adj)
            if layer_idx != len(self.conv_layers) - 1: 
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
        return x


class self_gating(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout_rate=0.0, temperature=1.0):
        super(self_gating, self).__init__()
        self.temperature = temperature
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, feature_tensor):
        attention_scores = self.gate_layer(feature_tensor) / self.temperature
        gating_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = (gating_weights * feature_tensor).sum(dim=1)
        return weighted_output, gating_weights

class RNA_Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True)
        })
        
        output_blocks = []
        for _ in range(3):  
            block = torch.nn.Sequential()
            block.add_module("proj", torch.nn.Linear(hidden_dim, out_dim))
            output_blocks.append(block)
        self.pi_head, self.disp_head, self.mean_head = output_blocks
        

        self.output_activations = torch.nn.ModuleDict({
            "disp": self._create_activation("softplus", 1e-4, 1e4),
            "mean": self._create_activation("exp", 1e-5, 1e6)
        })

    def _create_activation(self, func_type, min_val, max_val):
        class CustomActivation(torch.nn.Module):
            def forward(self, x):
                if func_type == "softplus":
                    output = F.softplus(x)
                elif func_type == "exp":
                    output = torch.exp(x)
                return torch.clamp(output, min_val, max_val)
        return CustomActivation()

    def forward(self, latent):

        features = self.feature_net["linear"](latent)
        features = self.feature_net["normalization"](features)
        processed = self.feature_net["activation"](features)

        pi_param = torch.sigmoid(self.pi_head(processed))
        disp_param = self.output_activations["disp"](self.disp_head(processed))
        mean_param = self.output_activations["mean"](self.mean_head(processed))
        
        return pi_param, disp_param, mean_param

class ATAC_Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True)
        })
        
        self.prob_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim)
        )
        
        self.output_activations = torch.nn.ModuleDict({
            "prob": torch.nn.Sigmoid() 
        })

    def forward(self, latent):
        hidden = self.feature_net["linear"](latent)
        normalized = self.feature_net["normalization"](hidden)
        processed = self.feature_net["activation"](normalized)

        logits = self.prob_head(processed)
        probabilities = self.output_activations["prob"](logits)
        
        return probabilities

class ADT_Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True)
        })
        
        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim)
        )
        
        self.output_activations = torch.nn.ModuleDict({
            "mean": torch.nn.Identity()  
        })

    def forward(self, latent):
        features = self.feature_net["linear"](latent)
        normalized = self.feature_net["normalization"](features)
        processed = self.feature_net["activation"](normalized)
        
        mean_param = self.output_activations["mean"](self.mean_head(processed))
        
        return mean_param

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True)
        })
        
        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim)
        )
        
        self.output_activations = torch.nn.ModuleDict({
            "mean": torch.nn.Identity()  
        })

    def forward(self, latent):
        features = self.feature_net["linear"](latent)
        normalized = self.feature_net["normalization"](features)
        processed = self.feature_net["activation"](normalized)
        
        mean_param = self.output_activations["mean"](self.mean_head(processed))
        
        return mean_param