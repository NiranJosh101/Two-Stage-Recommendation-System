from dataclasses import dataclass
from typing import Dict, Optional
import torch


class SchemaValidationError(Exception):
    """Raised when batch does not conform to expected embedding schema."""
    pass


@dataclass(frozen=True)
class TowerSchema:
    """
    Schema for pre-computed embedding towers.
    """
    feature_name: str     # e.g., "user_embedding" or "job_embedding"
    expected_dim: int     # e.g., 783 or 1159
    check_nan: bool = True

    def validate_batch(self, tensor: torch.Tensor) -> None:
        """
        Validate the embedding tensor for a batch.
        """
        # 1. Type Check
        if not torch.is_floating_point(tensor):
            raise SchemaValidationError(
                f"Feature '{self.feature_name}' must be a float tensor, got {tensor.dtype}"
            )

        # 2. Dimensionality Check
        # Expecting [batch_size, expected_dim]
        if tensor.ndim != 2:
            raise SchemaValidationError(
                f"Feature '{self.feature_name}' must be 2D [batch_size, dim], got {tensor.ndim}D"
            )

        actual_dim = tensor.shape[1]
        if actual_dim != self.expected_dim:
            raise SchemaValidationError(
                f"Dimension mismatch for '{self.feature_name}': "
                f"Expected {self.expected_dim}, got {actual_dim}"
            )

        # 3. Data Integrity (NaNs and Infs)
        if self.check_nan:
            if torch.isnan(tensor).any():
                raise SchemaValidationError(f"NaN detected in '{self.feature_name}' batch")
            if torch.isinf(tensor).any():
                raise SchemaValidationError(f"Inf detected in '{self.feature_name}' batch")

class TwoTowerValidator:
    """
    Coordinates validation for both towers simultaneously.
    """
    def __init__(self, user_schema: TowerSchema, job_schema: TowerSchema):
        self.user_schema = user_schema
        self.job_schema = job_schema

    def __call__(self, user_batch: torch.Tensor, job_batch: torch.Tensor):
        # Validate individual towers
        self.user_schema.validate_batch(user_batch)
        self.job_schema.validate_batch(job_batch)

        # Cross-tower consistency (Batch size must match)
        if user_batch.shape[0] != job_batch.shape[0]:
            raise SchemaValidationError(
                f"Batch size mismatch: User ({user_batch.shape[0]}) vs "
                f"Job ({job_batch.shape[0]})"
            )