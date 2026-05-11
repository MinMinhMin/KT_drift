"""
autoencoder.py - Refined Dual-AE with Orthogonality Loss & Explicit Latents
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, out_dim: int, hidden_dims: list[int], activation=nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

# ---------------------------------------------------------------------------
# KTAutoencoder (v2: Disentangled Branches)
# ---------------------------------------------------------------------------
class KTAutoencoder(nn.Module):
    def __init__(
        self, 
        state_dim: int = 33, 
        behavior_dim: int = 6, 
        z_s_dim: int = 16, 
        z_b_dim: int = 8, 
        lstm_hidden: int = 64
    ):
        
        super().__init__() 
        # Tổng input dim = 33 (state) + 6 (behavior) = 39

        self.state_dim = state_dim
        self.behavior_dim = behavior_dim
        self.z_s_dim = z_s_dim
        self.z_b_dim = z_b_dim

        # 1. Encoders
        # State: 33 -> 32 -> 16
        self.state_encoder = _mlp(state_dim, z_s_dim, [32])
        
        # Behavior: 6 -> 16 -> 8
        self.behavior_encoder = _mlp(behavior_dim, z_b_dim, [16])

        # 2. FiLM Modulation (gamma, beta)
        self.film_gamma = nn.Linear(z_s_dim, z_b_dim)
        self.film_beta = nn.Linear(z_s_dim, z_b_dim)

        # 3. Temporal Backbone (BiLSTM)
        # Input: concat(z_s, z_b_prime) = 16 + 8 = 24
        self.temporal_backbone = nn.LSTM(
            input_size=z_s_dim + z_b_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 4. Decoders
        # State decoder: z_s -> state_hat
        self.state_decoder = _mlp(z_s_dim, state_dim, [32])
        
        # Behavior decoder: z_b_prime -> behavior_hat
        self.behavior_decoder = _mlp(z_b_dim, behavior_dim, [16])

        # 5. Predictive Head
        # Input: h_t (lstm_hidden * 2) + z_s(t+1)
        self.predictor = _mlp(lstm_hidden * 2 + z_s_dim, 1, [32])

    def forward(self, x_sequence: torch.Tensor) -> dict:
        """
        x_sequence shape: [batch, seq_len, 39]
        """
        batch_size, seq_len, _ = x_sequence.shape

        # Split inputs
        x_s = x_sequence[:, :, :self.state_dim]          # [B, L, 33]
        x_b = x_sequence[:, :, self.state_dim:]         # [B, L, 6]

        # Step 1: Encoding
        z_s = self.state_encoder(x_s)                    # [B, L, 16]
        z_b = self.behavior_encoder(x_b)                 # [B, L, 8]

        # Step 2: FiLM modulation
        gamma = torch.sigmoid(self.film_gamma(z_s))
        beta = self.film_beta(z_s)

        z_b_prime = gamma * z_b + beta                 # [B, L, 8]

        # Step 3: Fusion & Temporal modeling
        z_fused = torch.cat([z_s, z_b_prime], dim=-1)    # [B, L, 24]
        h, _ = self.temporal_backbone(z_fused)           # [B, L, 128] (BiLSTM)

        # Step 4: Reconstruction
        x_s_hat = self.state_decoder(z_s)
        x_b_hat = self.behavior_decoder(z_b_prime)

        # Step 5: Prediction (a_t+1)
        # Shift temporal features to align h_t with z_s(t+1)
        # h_t: 0 to L-1, z_s_next: 1 to L
        h_current = h[:, :-1, :] 
        z_s_next = z_s[:, 1:, :]
        
        pred_input = torch.cat([h_current, z_s_next], dim=-1)
        y_hat = torch.sigmoid(self.predictor(pred_input)) # [B, L-1, 1]

        return {
            "z_s": z_s,
            "z_b": z_b,
            "z_b_prime": z_b_prime,
            "x_s_hat": x_s_hat,
            "x_b_hat": x_b_hat,
            "y_hat": y_hat
        }

    def compute_loss(
        self, 
        x_sequence: torch.Tensor, 
        output: dict, 
        alpha: float = 1.0, 
        lambda_pred: float = 1.0, 
        beta: float = 0.1
    ) -> dict:
        """
        L = L_state + alpha*L_behavior + lambda*L_pred + beta*L_sep
        """
        # 1. Reconstruction Loss
        x_s_target = x_sequence[:, :, :self.state_dim]
        x_b_target = x_sequence[:, :, self.state_dim:]
        
        l_state = F.mse_loss(output["x_s_hat"], x_s_target)
        
        # Behavior loss (BCE cho response - dim 0, MSE cho các phần còn lại)
        # Giả định: x_b[:, :, 0:2] là response (one-hot hoặc multi-label)
        l_behavior = F.mse_loss(output["x_b_hat"], x_b_target) 

        # 2. Predictive Loss
        # Target là response của bước tiếp theo (x_b_t+1)
        # Giả định response nằm ở index đầu tiên của behavior features
        y_target = x_sequence[:, 1:, self.state_dim : self.state_dim + 1]
        l_pred = F.binary_cross_entropy(output["y_hat"], y_target)

        # 3. Separation Loss (Normalized Cross-Covariance)

        z_s_flat = output["z_s"].reshape(-1, self.z_s_dim)   # [N, d_s]
        z_b_flat = output["z_b"].reshape(-1, self.z_b_dim)   # [N, d_b]

        # Centering (remove mean)
        z_s_centered = z_s_flat - z_s_flat.mean(dim=0, keepdim=True)
        z_b_centered = z_b_flat - z_b_flat.mean(dim=0, keepdim=True)

        # Normalize variance (optional but recommended)
        z_s_norm = z_s_centered / (z_s_centered.std(dim=0, keepdim=True) + 1e-6)
        z_b_norm = z_b_centered / (z_b_centered.std(dim=0, keepdim=True) + 1e-6)

        # Cross-correlation matrix
        N = z_s_flat.size(0)
        cross_corr = torch.matmul(z_s_norm.t(), z_b_norm) / N

        # Penalize correlation
        l_sep = torch.norm(cross_corr, p='fro')

        # Total
        total_loss = l_state + alpha * l_behavior + lambda_pred * l_pred + beta * l_sep

        return {
            "loss": total_loss,
            "l_state": l_state,
            "l_behavior": l_behavior,
            "l_pred": l_pred,
            "l_sep": l_sep
        }