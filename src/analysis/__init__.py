"""
Phase 6: Advanced Analysis Module
Multi-feature analysis platform for explainability, interaction, tracking, and cohort analysis.
"""

from .statistical_utils import (
    compute_significance,
    compute_effect_size,
    mann_whitney_u_test,
    compute_confidence_intervals
)

from .embedding_indexer import (
    EmbeddingIndexer,
    build_index_from_features
)

__all__ = [
    'compute_significance',
    'compute_effect_size',
    'mann_whitney_u_test',
    'compute_confidence_intervals',
    'EmbeddingIndexer',
    'build_index_from_features'
]
