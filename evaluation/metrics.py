"""
Evaluation Metrics
PSR, F1, Precision, Recall, Calibration, Cognitive Coverage
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class EvaluationMetrics:
    """Comprehensive evaluation metrics for EduGen"""
    
    @staticmethod
    def pedagogical_success_rate(predictions: np.ndarray, 
                                 factual_checks: np.ndarray,
                                 bloom_checks: np.ndarray) -> float:
        """
        Compute Pedagogical Success Rate (PSR).
        PSR = % of questions that are factually correct AND bloom-aligned
        """
        valid = (factual_checks == 1) & (bloom_checks == 1)
        return float(valid.sum() / len(predictions))
    
    @staticmethod
    def compute_calibration_error(confidences: np.ndarray,
                                  correctness: np.ndarray,
                                  n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
            if bin_mask.sum() > 0:
                bin_acc = correctness[bin_mask].mean()
                bin_conf = confidences[bin_mask].mean()
                ece += np.abs(bin_acc - bin_conf) * (bin_mask.sum() / len(confidences))
        
        return float(ece)
    
    @staticmethod
    def cognitive_coverage(bloom_levels: List[int]) -> Dict:
        """Compute distribution across Bloom levels"""
        unique, counts = np.unique(bloom_levels, return_counts=True)
        total = len(bloom_levels)
        return {
            'distribution': {int(k): int(v) for k, v in zip(unique, counts)},
            'percentages': {int(k): float(v/total*100) for k, v in zip(unique, counts)}
        }
    
    @staticmethod
    def hallucination_rate(factual_checks: np.ndarray) -> float:
        """Compute hallucination rate (factual errors)"""
        return float((factual_checks == 0).sum() / len(factual_checks))
EOF