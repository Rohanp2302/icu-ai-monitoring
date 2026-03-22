"""
Phase 5: Explainability Module

Comprehensive model interpretability tools:
- SHAP feature importance
- Attention pattern extraction
- Clinical rule generation
- Unified clinical interpreter
"""

from .shap_explainer import SHAPExplainer, AttentionExplainer, FEATURE_NAMES
from .rule_extractor import RuleExtractor, ClinicalRule
from .clinical_interpreter import ClinicalInterpreter

__all__ = [
    'SHAPExplainer',
    'AttentionExplainer',
    'RuleExtractor',
    'ClinicalRule',
    'ClinicalInterpreter',
    'FEATURE_NAMES',
]
