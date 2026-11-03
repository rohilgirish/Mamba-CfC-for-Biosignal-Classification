import torch
import torch.nn as nn
from ncps.torch import CfC
from src.models.mamba_fallback import Mamba, MambaConfig

class PoolingDB5(nn.Module):
    def __init__(self, mode='last'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        # x shape: (B, L, D)
        if self.mode == 'last':
            return x[:, -1, :]
        elif self.mode == 'mean':
            return x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")

class MambaOnlyModelDB5(nn.Module):
    def __init__(self, input_dim=16, num_classes=53, d_model=64, num_layers=2, pool_mode='last', expand_factor=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mamba_backbone = Mamba(MambaConfig(d_model=d_model, n_layers=num_layers, expand_factor=expand_factor))
        self.pooling = PoolingDB5(mode=pool_mode)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba_backbone(x)
        x = self.pooling(x)
        return self.head(x)

class CfCOnlyModelDB5(nn.Module):
    def __init__(self, input_dim=16, num_classes=53, d_model=64, pool_mode='last'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.cfc = CfC(input_size=d_model, units=64, mode='default', return_sequences=True, batch_first=True)
        self.pooling = PoolingDB5(mode=pool_mode)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.cfc(x)
        x = self.pooling(x)
        return self.head(x)

class MambaCfCModelDB5(nn.Module):
    def __init__(self, input_dim=16, num_classes=53, d_model=64, num_layers=2, cfc_units=64, pool_mode='last'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mamba_backbone = Mamba(MambaConfig(d_model=d_model, n_layers=num_layers))
        self.projection = nn.Linear(d_model, cfc_units)
        self.cfc = CfC(input_size=cfc_units, units=cfc_units, mode='default', return_sequences=True, batch_first=True)
        self.pooling = PoolingDB5(mode=pool_mode)
        self.head = nn.Linear(cfc_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba_backbone(x)
        x = self.projection(x)
        x, _ = self.cfc(x)
        x = self.pooling(x)
        return self.head(x)

def get_model_db5(model_name: str, pool_mode: str = 'last', device: str = 'cpu'):
    input_dim = 16
    num_classes = 53
    
    if model_name == 'mamba-only':
        model = MambaOnlyModelDB5(input_dim=input_dim, num_classes=num_classes, d_model=64, num_layers=2, pool_mode=pool_mode)
    elif model_name == 'cfc-only':
        model = CfCOnlyModelDB5(input_dim=input_dim, num_classes=num_classes, d_model=64, pool_mode=pool_mode)
    elif model_name == 'mamba-cfc':
        model = MambaCfCModelDB5(input_dim=input_dim, num_classes=num_classes, d_model=64, num_layers=2, cfc_units=64, pool_mode=pool_mode)
    elif model_name == 'mamba-3l':
        model = MambaOnlyModelDB5(input_dim=input_dim, num_classes=num_classes, d_model=64, num_layers=3, pool_mode=pool_mode)
    elif model_name == 'mamba-width':
        model = MambaOnlyModelDB5(input_dim=input_dim, num_classes=num_classes, d_model=88, num_layers=2, pool_mode=pool_mode)
    else:
        raise ValueError(f"Unknown model name for DB5: {model_name}")
        
    return model.to(device)

