import torch.nn as nn

class QWaveModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64,1)
            
        )
        
    def forward(self, X):
        return self.model(X)