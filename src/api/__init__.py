"""
API Module - REST endpoints for ICU model predictions and explanations
"""

from .explainability_api import ExplainabilityAPI, create_api_app

__all__ = ['ExplainabilityAPI', 'create_api_app']
