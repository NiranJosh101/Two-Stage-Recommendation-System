import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    """
    Processes pre-computed embeddings of a specific dimension into a shared space.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        
        # Iteratively build the MLP
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # Using BatchNorm for stable training
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1)) # Added dropout to prevent overfitting
            curr_dim = h_dim
            
        self.mlp = nn.Sequential(*layers)
        self.final_projection = nn.Linear(curr_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_dim]
        x = self.mlp(x)
        return self.final_projection(x)

class TwoTowerModel(nn.Module):
    """
    Retrieval model for asymmetric input dimensions.
    """
    def __init__(
        self,
        user_dim: int,
        job_dim: int,
        hidden_dims: list[int] = [512, 256],
        output_dim: int = 128,
    ):
        super().__init__()

        self.user_tower = Tower(user_dim, hidden_dims, output_dim)
        self.job_tower = Tower(job_dim, hidden_dims, output_dim)

    def forward(self, user_features: torch.Tensor, job_features: torch.Tensor) -> torch.Tensor:
        # 1. Generate representations
        user_vector = self.user_tower(user_features)
        job_vector = self.job_tower(job_features)

        # 2. L2 Normalize for Cosine Similarity (standard for retrieval)
        user_vector = F.normalize(user_vector, p=2, dim=1)
        job_vector = F.normalize(job_vector, p=2, dim=1)

        # 3. Compute similarity (dot product of normalized vectors)
        # Returns shape [batch_size]
        return torch.sum(user_vector * job_vector, dim=1)