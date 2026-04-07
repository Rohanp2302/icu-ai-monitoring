"""
Phase 3: Multi-Task Deep Learning Model

Architecture:
- Shared Transformer Encoder: 24 timesteps × 42 features → contextual embedding
- 5 Task-Specific Decoders:
  1. Mortality (binary classification, AUC > 0.85)
  2. Risk Stratification (4-class, F1 > 0.72)
  3. Clinical Outcomes (multi-label: sepsis, AKI, ARDS, shock, MODS, ARF)
  4. Treatment Response (vitals deviation from targets)
  5. LOS Prediction (total, remaining, discharge probability)

Features:
- Multi-head attention for temporal reasoning
- MC Dropout for uncertainty estimation
- Task-specific loss weighting
- Batch normalization throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import json


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer (fixed sinusoidal)"""

    def __init__(self, d_model: int, max_seq_length: int = 24, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Temporal encoder using multi-head attention"""

    def __init__(
        self,
        input_dim: int = 42,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_seq_length: int = 24,
    ):
        super(TransformerEncoder, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, seq_len, d_model)
        """
        # Project input to d_model
        x = self.input_projection(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)  # (B, T, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T, d_model)

        return x


class StaticFeatureEncoder(nn.Module):
    """Encode static demographic/comorbidity features"""

    def __init__(self, static_dim: int = 20, output_dim: int = 128, dropout: float = 0.3):
        super(StaticFeatureEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(static_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, static_dim)
        Returns:
            (batch_size, output_dim)
        """
        return self.network(x)


class TemporalPooling(nn.Module):
    """Global temporal pooling with attention"""

    def __init__(self, d_model: int = 256):
        super(TemporalPooling, self).__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, d_model)
        """
        # Compute attention weights
        attn_weights = F.softmax(self.attention(x), dim=1)  # (B, T, 1)

        # Weighted average
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, d_model)
        return pooled


class ModalityGatedFusion(nn.Module):
    """Learned gating between temporal and static modality embeddings."""

    def __init__(self, temporal_dim: int, static_dim: int, hidden_dim: int = 64):
        super(ModalityGatedFusion, self).__init__()
        self.temporal_dim = temporal_dim
        self.static_dim = static_dim
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.norm = nn.LayerNorm(temporal_dim + static_dim)

    def forward(self, temporal: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([temporal, static], dim=1)
        gate_logits = self.gate(concat)
        weights = torch.softmax(gate_logits, dim=1)

        weighted = torch.cat(
            [
                temporal * weights[:, 0:1],
                static * weights[:, 1:2],
            ],
            dim=1,
        )

        # Residual stabilization keeps base representation while learning modality preference.
        fused = self.norm(weighted + 0.5 * concat)
        return fused


class MortalityDecoder(nn.Module):
    """Binary mortality prediction decoder with MC Dropout"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.3):
        super(MortalityDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Binary output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, 1) logits
        """
        return self.network(x)


class RiskStratificationDecoder(nn.Module):
    """4-class risk stratification decoder"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.3):
        super(RiskStratificationDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),  # 4 risk classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, 4) class logits
        """
        return self.network(x)


class ClinicalOutcomesDecoder(nn.Module):
    """Multi-label clinical outcomes decoder (sepsis, AKI, ARDS, shock, MODS, ARF)"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, n_outcomes: int = 6, dropout: float = 0.3):
        super(ClinicalOutcomesDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outcomes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, n_outcomes) independent logits
        """
        return self.network(x)


class TreatmentResponseDecoder(nn.Module):
    """Predict deviation from therapeutic targets [HR, RR, SaO2]"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.3):
        super(TreatmentResponseDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # Deviation from target for HR, RR, SaO2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, 3) therapeutic deviations (MSE loss)
        """
        return self.network(x)


class LOSPredictionDecoder(nn.Module):
    """Predict length of stay: total_los, remaining_los, discharge_probability"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.3):
        super(LOSPredictionDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Three separate output heads
        self.total_los_head = nn.Linear(64, 1)  # Positive constraint via softplus
        self.remaining_los_head = nn.Linear(64, 1)  # Positive constraint
        self.discharge_prob_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            Tuple of:
            - total_los: (batch_size, 1) days (positive)
            - remaining_los: (batch_size, 1) days (positive)
            - discharge_prob: (batch_size, 1) probability [0, 1]
        """
        features = self.network(x)

        # Total LOS: positive via softplus
        total_los = F.softplus(self.total_los_head(features))

        # Remaining LOS: positive via softplus
        remaining_los = F.softplus(self.remaining_los_head(features))

        # Discharge probability: [0, 1] via sigmoid
        discharge_prob = self.discharge_prob_head(features)

        return total_los, remaining_los, discharge_prob


class MultiTaskICUModel(nn.Module):
    """Unified multi-task ICU prediction model"""

    def __init__(
        self,
        input_dim: int = 42,
        static_dim: int = 20,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        static_output_dim: int = 128,
        dropout: float = 0.3,
        n_outcomes: int = 6,
        modality_dropout_prob: float = 0.1,
    ):
        super(MultiTaskICUModel, self).__init__()

        # Shared components
        self.temporal_encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.temporal_pooling = TemporalPooling(d_model)

        self.static_encoder = StaticFeatureEncoder(
            static_dim=static_dim, output_dim=static_output_dim, dropout=dropout
        )

        # Combined embedding dimension
        combined_dim = d_model + static_output_dim
        self.modality_dropout_prob = modality_dropout_prob
        self.fusion = ModalityGatedFusion(d_model, static_output_dim)

        # Task-specific decoders
        self.mortality_decoder = MortalityDecoder(input_dim=combined_dim, dropout=dropout)
        self.risk_decoder = RiskStratificationDecoder(input_dim=combined_dim, dropout=dropout)
        self.outcomes_decoder = ClinicalOutcomesDecoder(
            input_dim=combined_dim, n_outcomes=n_outcomes, dropout=dropout
        )
        self.response_decoder = TreatmentResponseDecoder(input_dim=combined_dim, dropout=dropout)
        self.los_decoder = LOSPredictionDecoder(input_dim=combined_dim, dropout=dropout)

        # Task-specific loss weights (learnable)
        self.log_task_weights = nn.Parameter(torch.zeros(5))

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_static: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all tasks.

        Args:
            x_temporal: (batch_size, seq_len, input_dim) engineered features
            x_static: (batch_size, static_dim) demographic/comorbidity features

        Returns:
            Dict with outputs for all 5 tasks
        """
        # Encode temporal features
        temporal_encoded = self.temporal_encoder(x_temporal)  # (B, T, d_model)
        temporal_pooled = self.temporal_pooling(temporal_encoded)  # (B, d_model)

        # Encode static features
        if x_static is None:
            static_dim = self.static_encoder.network[0].in_features
            x_static = torch.zeros(
                x_temporal.size(0),
                static_dim,
                dtype=x_temporal.dtype,
                device=x_temporal.device,
            )
        static_encoded = self.static_encoder(x_static)  # (B, static_output_dim)

        # Stochastic modality dropout during training to improve robustness.
        if self.training and self.modality_dropout_prob > 0:
            if torch.rand(1, device=temporal_pooled.device).item() < self.modality_dropout_prob:
                if torch.rand(1, device=temporal_pooled.device).item() < 0.5:
                    temporal_pooled = torch.zeros_like(temporal_pooled)
                else:
                    static_encoded = torch.zeros_like(static_encoded)

        # Learn modality weighting before task decoding.
        combined = self.fusion(temporal_pooled, static_encoded)

        # Task predictions
        outputs = {
            "mortality": self.mortality_decoder(combined),  # (B, 1)
            "risk": self.risk_decoder(combined),  # (B, 4)
            "outcomes": self.outcomes_decoder(combined),  # (B, 6)
            "response": self.response_decoder(combined),  # (B, 3)
        }

        # LOS has 3 outputs
        total_los, remaining_los, discharge_prob = self.los_decoder(combined)
        outputs["total_los"] = total_los  # (B, 1)
        outputs["remaining_los"] = remaining_los  # (B, 1)
        outputs["discharge_prob"] = discharge_prob  # (B, 1)

        return outputs

    def get_task_weights(self) -> Dict[str, float]:
        """Get normalized task weights from learnable parameters"""
        weights = torch.softmax(self.log_task_weights, dim=0)
        task_names = ["mortality", "risk", "outcomes", "response", "los"]
        return {name: w.item() for name, w in zip(task_names, weights)}

    def freeze_encoder(self):
        """Freeze encoder for transfer learning"""
        for param in self.temporal_encoder.parameters():
            param.requires_grad = False
        for param in self.static_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder"""
        for param in self.temporal_encoder.parameters():
            param.requires_grad = True
        for param in self.static_encoder.parameters():
            param.requires_grad = True


class MultiTaskLoss(nn.Module):
    """Combined loss for all 5 prediction tasks"""

    def __init__(self, device="cpu", use_focal_for_mortality: bool = True, focal_gamma: float = 2.0, focal_alpha: float = 0.75):
        super(MultiTaskLoss, self).__init__()
        self.device = device
        self.use_focal_for_mortality = use_focal_for_mortality
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Task-specific criteria
        self.mortality_loss = nn.BCEWithLogitsLoss()  # Binary cross-entropy (logits)
        self.risk_loss = nn.CrossEntropyLoss()  # Categorical cross-entropy
        self.outcomes_loss = nn.BCEWithLogitsLoss()  # Multi-label BCE (logits)
        self.response_loss = nn.MSELoss()  # Mean squared error
        self.los_loss = nn.SmoothL1Loss()  # Smooth L1 for count regression

    def _focal_bce_with_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary focal loss on logits for class-imbalanced mortality prediction."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - pt) ** self.focal_gamma) * bce
        return focal.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss.

        Args:
            outputs: Model predictions (from forward pass)
            targets: Ground truth labels
            task_weights: (5,) tensor of normalized weights

        Returns:
            total_loss: Scalar loss
            loss_dict: Dict of individual task losses
        """
        loss_dict = {}

        # Mortality loss (binary)
        if "mortality" in targets:
            mortality_targets = targets["mortality"].unsqueeze(1).float()
            if self.use_focal_for_mortality:
                loss_dict["mortality"] = self._focal_bce_with_logits(
                    outputs["mortality"], mortality_targets
                )
            else:
                loss_dict["mortality"] = self.mortality_loss(
                    outputs["mortality"], mortality_targets
                )

        # Risk stratification loss (4-class)
        if "risk" in targets:
            loss_dict["risk"] = self.risk_loss(
                outputs["risk"], targets["risk"].long().view(-1)
            )

        # Clinical outcomes loss (multi-label)
        if "outcomes" in targets:
            loss_dict["outcomes"] = self.outcomes_loss(
                outputs["outcomes"], targets["outcomes"].float()
            )

        # Treatment response loss
        if "response" in targets:
            loss_dict["response"] = self.response_loss(
                outputs["response"], targets["response"].float()
            )

        # LOS prediction loss
        if "los" in targets:
            los_total = targets["los"][:, 0]  # First column: total LOS
            los_remaining = targets["los"][:, 1]  # Second column: remaining LOS

            loss_total_los = self.los_loss(
                outputs["total_los"].squeeze(), los_total.float()
            )
            loss_remaining_los = self.los_loss(
                outputs["remaining_los"].squeeze(), los_remaining.float()
            )
            loss_dict["los"] = (loss_total_los + loss_remaining_los) / 2

        # Weighted sum
        total_loss = torch.tensor(0.0, device=self.device)
        task_names = ["mortality", "risk", "outcomes", "response", "los"]

        for i, task_name in enumerate(task_names):
            if task_name in loss_dict:
                total_loss += task_weights[i] * loss_dict[task_name]

        return total_loss, loss_dict


def create_model(
    input_dim: int = 42,
    static_dim: int = 20,
    device: str = "cpu",
) -> Tuple[MultiTaskICUModel, MultiTaskLoss]:
    """
    Create multi-task model and loss function.

    Args:
        input_dim: Temporal feature dimension (42 from feature engineering)
        static_dim: Static feature dimension (20 for demographics)
        device: 'cpu' or 'cuda'

    Returns:
        model: MultiTaskICUModel instance
        criterion: MultiTaskLoss instance
    """
    model = MultiTaskICUModel(
        input_dim=input_dim,
        static_dim=static_dim,
        d_model=256,
        n_heads=8,
        n_layers=3,
        dim_feedforward=512,
        static_output_dim=128,
        dropout=0.3,
        n_outcomes=6,
    ).to(device)

    criterion = MultiTaskLoss(device=device)

    return model, criterion


if __name__ == "__main__":
    # Quick test
    print("=" * 80)
    print("PHASE 3: MULTI-TASK DEEP LEARNING MODEL - TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model, criterion = create_model(device=device)
    print(f"Model created successfully")

    # Print model architecture
    print(f"\nModel Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 32
    seq_len = 24
    input_dim = 42
    static_dim = 20

    x_temporal = torch.randn(batch_size, seq_len, input_dim).to(device)
    x_static = torch.randn(batch_size, static_dim).to(device)

    outputs = model(x_temporal, x_static)

    print(f"\nOutput shapes:")
    for task, output in outputs.items():
        print(f"  {task:20} {output.shape}")

    # Convert logits to probabilities for preview.
    output_probs = {
        "mortality": torch.sigmoid(outputs["mortality"]),
        "risk": torch.softmax(outputs["risk"], dim=1),
        "outcomes": torch.sigmoid(outputs["outcomes"]),
        "response": outputs["response"],
        "total_los": outputs["total_los"],
        "remaining_los": outputs["remaining_los"],
        "discharge_prob": outputs["discharge_prob"],
    }
    print(f"\nProbability preview:")
    print(f"  mortality mean prob: {output_probs['mortality'].mean().item():.4f}")
    print(f"  risk row sums (first 3): {output_probs['risk'][:3].sum(dim=1).detach().cpu().numpy()}")

    # Test loss computation
    print(f"\nTesting loss computation...")
    targets = {
        "mortality": torch.randint(0, 2, (batch_size,)).to(device),
        "risk": torch.randint(0, 4, (batch_size,)).to(device),
        "outcomes": torch.randint(0, 2, (batch_size, 6)).to(device),
        "response": torch.randn(batch_size, 3).to(device),
        "los": torch.abs(torch.randn(batch_size, 2)).to(device),
    }

    task_weights = torch.ones(5).to(device) / 5
    total_loss, loss_dict = criterion(outputs, targets, task_weights)

    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Individual losses:")
    for task_name, task_loss in loss_dict.items():
        print(f"    {task_name:20} {task_loss.item():.4f}")

    print(f"\nTask weight distribution:")
    task_weights_dict = model.get_task_weights()
    for task_name, weight in task_weights_dict.items():
        print(f"  {task_name:20} {weight:.4f}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Multi-task model architecture complete and tested")
    print("=" * 80)
