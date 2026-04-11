"""
Week 2 Phase 2: LSTM Checkpoint Evaluation
Load checkpoints and benchmark performance on temporal data
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, f1_score,
    confusion_matrix, accuracy_score,
    precision_score, recall_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMCheckpointEvaluator:
    """Load and evaluate pre-trained LSTM checkpoints"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.checkpoint_dir = self.project_root / 'checkpoints' / 'multimodal'
        self.data_dir = self.project_root / 'data'
        self.results_dir = self.project_root / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def load_temporal_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load extracted temporal sequences"""
        logger.info("\n" + "="*80)
        logger.info("LOADING TEMPORAL DATA")
        logger.info("="*80)
        
        X_24h = np.load(self.data_dir / 'X_24h.npy')
        X_static = np.load(self.data_dir / 'X_static_24h.npy')
        y_24h = np.load(self.data_dir / 'y_24h.npy')
        
        logger.info(f"✓ X_24h: {X_24h.shape}")
        logger.info(f"✓ X_static: {X_static.shape}")
        logger.info(f"✓ y_24h: {y_24h.shape}")
        logger.info(f"  Mortality rate: {np.mean(y_24h):.1%}")
        
        return X_24h, X_static, y_24h
    
    def load_lstm_checkpoint(self, fold_idx: int) -> Optional[torch.nn.Module]:
        """Load LSTM model from checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'fold_{fold_idx}_best_model.pt'
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            from src.models.multitask_model import MultiTaskICUModel
            
            logger.info(f"\nLoading checkpoint: fold_{fold_idx}")
            
            # Create model with CORRECT checkpoint architecture
            model = MultiTaskICUModel(
                input_dim=6,           # temporal features
                static_dim=8,          # static features
                d_model=320,           # embedding dimension
                n_heads=8,
                n_layers=3,
                dim_feedforward=512,
                static_output_dim=128,
                dropout=0.3,
                n_outcomes=6,
            ).to(self.device)
            
            # Load weights
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            logger.info(f"✓ Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def run_inference(self, model: torch.nn.Module, 
                     X_temporal: np.ndarray, 
                     X_static: np.ndarray,
                     batch_size: int = 32) -> np.ndarray:
        """Run inference on data"""
        
        # Normalize static features to prevent NaN issues
        X_static_norm = (X_static - np.mean(X_static, axis=0)) / (np.std(X_static, axis=0) + 1e-8)
        X_static_norm = np.clip(X_static_norm, -3, 3)
        X_static_norm = np.nan_to_num(X_static_norm, nan=0.0)
        
        # Normalize temporal features
        X_temporal_reshape = X_temporal.reshape(-1, X_temporal.shape[-1])
        X_temporal_norm = (X_temporal_reshape - np.mean(X_temporal_reshape, axis=0)) / (np.std(X_temporal_reshape, axis=0) + 1e-8)
        X_temporal_norm = np.clip(X_temporal_norm, -3, 3)
        X_temporal_norm = X_temporal_norm.reshape(X_temporal.shape)
        X_temporal_norm = np.nan_to_num(X_temporal_norm, nan=0.0)
        
        y_pred_prob = []
        
        with torch.no_grad():
            for i in range(0, len(X_temporal_norm), batch_size):
                x_t = torch.FloatTensor(X_temporal_norm[i:i+batch_size]).to(self.device)
                x_s = torch.FloatTensor(X_static_norm[i:i+batch_size]).to(self.device)
                
                outputs = model(x_t, x_s)
                mortality_logits = outputs['mortality'].squeeze(-1)  # (batch,)
                mortality_prob = torch.sigmoid(mortality_logits)
                
                # Handle NaN/Inf in probabilities
                prob_np = mortality_prob.cpu().numpy()
                prob_np = np.nan_to_num(prob_np, nan=0.5, posinf=1.0, neginf=0.0)
                prob_np = np.clip(prob_np, 0.0, 1.0)
                
                y_pred_prob.extend(prob_np)
        
        return np.array(y_pred_prob)
    
    def compute_metrics(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        
        # Clean predictions
        y_pred_prob = np.nan_to_num(y_pred_prob, nan=0.5)
        y_pred_prob = np.clip(y_pred_prob, 0.0, 1.0)
        
        # Check for valid data
        if np.all(np.isnan(y_pred_prob)) or len(np.unique(y_true)) == 1:
            logger.warning("Invalid prediction data - using default metrics")
            return {
                'auc': 0.5,
                'optimal_threshold': 0.5,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'specificity': 0.0,
                'sensitivity': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'tn': 0,
                'total_deaths': int(np.sum(y_true)),
                'deaths_caught': 0,
            }
        
        # ROC-AUC
        try:
            auc_score = roc_auc_score(y_true, y_pred_prob)
        except Exception as e:
            logger.warning(f"AUC calculation failed: {e}, using 0.5")
            auc_score = 0.5
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Apply threshold
        y_pred_binary = (y_pred_prob >= optimal_threshold).astype(int)
        
        # Metrics
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        metrics = {
            'auc': float(auc_score),
            'optimal_threshold': float(optimal_threshold),
            'accuracy': float(accuracy_score(y_true, y_pred_binary)),
            'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred_binary, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_deaths': int(np.sum(y_true)),
            'deaths_caught': int(tp),
        }
        
        return metrics
    
    def evaluate_fold(self, fold_idx: int, 
                     X_temporal: np.ndarray,
                     X_static: np.ndarray,
                     y_24h: np.ndarray) -> Optional[Dict]:
        """Evaluate single fold"""
        
        logger.info("\n" + "─"*80)
        logger.info(f"FOLD {fold_idx} EVALUATION")
        logger.info("─"*80)
        
        # Load model
        model = self.load_lstm_checkpoint(fold_idx)
        if model is None:
            return None
        
        # Run inference
        logger.info("Running inference...")
        y_pred_prob = self.run_inference(model, X_temporal, X_static)
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self.compute_metrics(y_24h, y_pred_prob)
        
        # Log results
        logger.info(f"\nResults:")
        logger.info(f"  AUC:        {metrics['auc']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
        logger.info(f"  Threshold:  {metrics['optimal_threshold']:.4f}")
        logger.info(f"  Deaths caught: {metrics['deaths_caught']}/{metrics['total_deaths']}")
        
        return metrics
    
    def load_baseline_metrics(self) -> Dict:
        """Load RF baseline from Week 1"""
        logger.info("\n" + "="*80)
        logger.info("LOADING WEEK 1 BASELINE METRICS")
        logger.info("="*80)
        
        baseline_path = self.results_dir / 'threshold_summary.json'
        
        if baseline_path.exists():
            with open(baseline_path) as f:
                summary = json.load(f)
                result = summary.get('results', {}).get('f1', {})
                
                baseline = {
                    'auc': 0.8384,  # Known from RF evaluation
                    'recall': result.get('sensitivity', 0.721),
                    'f1': result.get('f1', 0.482),
                    'threshold': result.get('threshold', 0.44),
                    'deaths_caught': result.get('deaths_caught', 246),
                    'total_deaths': result.get('total_deaths', 341),
                }
                
                logger.info(f"✓ Loaded baseline")
                logger.info(f"  AUC: {baseline['auc']:.4f}")
                logger.info(f"  Recall: {baseline['recall']:.4f}")
                logger.info(f"  F1: {baseline['f1']:.4f}")
                
                return baseline
        
        logger.warning("Baseline file not found, using defaults")
        return {
            'auc': 0.8384,
            'recall': 0.721,
            'f1': 0.482,
            'threshold': 0.44,
            'deaths_caught': 246,
            'total_deaths': 341,
        }
    
    def run(self) -> bool:
        """Execute full evaluation pipeline"""
        
        logger.info("\n" + "█"*80)
        logger.info("█" + " "*78 + "█")
        logger.info("█" + "  WEEK 2 PHASE 2: LSTM CHECKPOINT EVALUATION".center(78) + "█")
        logger.info("█" + " "*78 + "█")
        logger.info("█"*80)
        
        try:
            # Load data
            X_24h, X_static, y_24h = self.load_temporal_data()
            
            # Load baseline
            baseline = self.load_baseline_metrics()
            
            # Evaluate all folds
            results = {
                'timestamp': str(np.datetime64('now')),
                'baseline': baseline,
                'folds': {}
            }
            
            best_fold = None
            best_auc = baseline['auc']
            
            for fold_idx in range(5):
                metrics = self.evaluate_fold(fold_idx, X_24h, X_static, y_24h)
                
                if metrics is not None:
                    results['folds'][f'fold_{fold_idx}'] = metrics
                    
                    if metrics['auc'] > best_auc:
                        best_auc = metrics['auc']
                        best_fold = fold_idx
            
            # Determine recommendation
            logger.info("\n" + "="*80)
            logger.info("EVALUATION COMPLETE - RECOMMENDATION")
            logger.info("="*80)
            
            if best_fold is not None:
                best_metrics = results['folds'][f'fold_{best_fold}']
                improvement = best_metrics['auc'] - baseline['auc']
                
                logger.info(f"\nBest performing fold: {best_fold}")
                logger.info(f"  AUC: {best_metrics['auc']:.4f} ({improvement:+.4f} vs baseline)")
                logger.info(f"  Recall: {best_metrics['recall']:.4f}")
                logger.info(f"  F1: {best_metrics['f1']:.4f}")
                
                if best_metrics['auc'] >= 0.86 and best_metrics['recall'] >= 0.75:
                    recommendation = "DEPLOY_LSTM"
                    reason = "Superior performance - research quality"
                elif best_metrics['auc'] >= baseline['auc']:
                    recommendation = "DEPLOY_ENSEMBLE"
                    reason = "Comparable/slight improvement - ensemble for robustness"
                else:
                    recommendation = "KEEP_RF"
                    reason = "RF baseline performs better"
                
                logger.info(f"\nRecommendation: {recommendation}")
                logger.info(f"Reason: {reason}")
                
                results['recommendation'] = {
                    'action': recommendation,
                    'reason': reason,
                    'best_fold': best_fold,
                    'best_metrics': best_metrics
                }
            else:
                logger.warning("No folds evaluated successfully")
                results['recommendation'] = {
                    'action': 'KEEP_RF',
                    'reason': 'LSTM evaluation failed'
                }
            
            # Save results
            output_path = self.results_dir / 'lstm_evaluation_results.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"\n✓ Results saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    evaluator = LSTMCheckpointEvaluator()
    success = evaluator.run()
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
