"""
Benchmark and Scholarship Suitability Computation Algorithm (BSSCA)
Implements Algorithm 7 from the paper

Computes standardized benchmark scores and scholarship suitability indices
for learner ranking and academic decision support.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from config import BSSCA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkScore:
    """
    Benchmark and scholarship scoring results.
    """
    learner_id: str
    benchmark_index: float
    scholarship_score: float
    rank_percentile: float
    performance_composite: float
    behavioral_stability: float
    cluster_advantage: float
    latent_strength: float
    scholarship_eligible: bool
    performance_band: str
    
    def __repr__(self):
        return (f"BenchmarkScore(id={self.learner_id}, "
                f"benchmark={self.benchmark_index:.3f}, "
                f"scholarship={self.scholarship_score:.3f}, "
                f"rank={self.rank_percentile:.1f}%, "
                f"eligible={self.scholarship_eligible})")


class BenchmarkScholarshipComputation:
    """
    BSSCA: Benchmark and Scholarship Suitability Computation Algorithm.
    
    Implements Algorithm 7 from the paper:
    1. Compute performance composite α_i
    2. Compute behavioral stability β_i
    3. Compute cluster advantage γ_i
    4. Compute latent strength δ_i
    5. Calculate benchmark index B_i
    6. Calculate scholarship score S_i
    7. Rank and classify learners
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize BSSCA.
        
        Args:
            config: Configuration dictionary. Uses BSSCA_CONFIG if None.
        """
        if config is None:
            config = BSSCA_CONFIG
        
        self.config = config
        self.benchmark_weights = config['benchmark_weights']
        self.scholarship_weights = config['scholarship_weights']
        self.cluster_advantages = config['cluster_advantages']
        self.scholarship_threshold = config['scholarship_threshold']
        
        logger.info("BSSCA initialized")
        logger.info(f"  Benchmark weights: {self.benchmark_weights}")
        logger.info(f"  Scholarship weights: {self.scholarship_weights}")
        logger.info(f"  Scholarship threshold: {self.scholarship_threshold}")
    
    def compute_benchmark_score(self,
                                learner_id: str,
                                profiling_outputs: Dict,
                                theta: float,
                                consistency_index: float) -> BenchmarkScore:
        """
        Compute benchmark and scholarship scores for a learner.
        
        Implements Algorithm 7 (BSSCA).
        
        Args:
            learner_id: Unique learner identifier
            profiling_outputs: Outputs from profiling models (PPE, PREM, LBSM, LLPE)
            theta: IRT ability estimate
            consistency_index: Performance consistency measure
            
        Returns:
            BenchmarkScore object
        """
        logger.debug(f"Computing benchmark score for learner {learner_id}")
        
        # Step 1: Compute performance composite α_i
        alpha = self._compute_performance_composite(profiling_outputs, theta)
        
        # Step 2: Compute behavioral stability β_i
        beta = self._compute_behavioral_stability(consistency_index)
        
        # Step 3: Compute cluster advantage γ_i
        gamma = self._compute_cluster_advantage(profiling_outputs)
        
        # Step 4: Compute latent strength δ_i
        delta = self._compute_latent_strength(profiling_outputs)
        
        # Step 5: Calculate benchmark index B_i
        benchmark_index = self._calculate_benchmark_index(alpha, beta, gamma, delta)
        
        # Step 6: Calculate scholarship score S_i
        scholarship_score = self._calculate_scholarship_score(benchmark_index, theta, delta)
        
        # Step 7: Determine eligibility and classification
        scholarship_eligible = scholarship_score >= self.scholarship_threshold
        performance_band = self._classify_performance(benchmark_index)
        
        # Create score object
        score = BenchmarkScore(
            learner_id=learner_id,
            benchmark_index=float(benchmark_index),
            scholarship_score=float(scholarship_score),
            rank_percentile=0.0,  # Will be computed later in batch processing
            performance_composite=float(alpha),
            behavioral_stability=float(beta),
            cluster_advantage=float(gamma),
            latent_strength=float(delta),
            scholarship_eligible=bool(scholarship_eligible),
            performance_band=performance_band
        )
        
        logger.debug(f"Benchmark score computed: {score}")
        
        return score
    
    def _compute_performance_composite(self,
                                      profiling_outputs: Dict,
                                      theta: float) -> float:
        """
        Compute performance composite α_i.
        
        Corresponds to line 1 of Algorithm 7: α_i = λ1*p_i + λ2*θ_i
        
        Args:
            profiling_outputs: Profiling model outputs
            theta: IRT ability estimate
            
        Returns:
            Performance composite score
        """
        # Extract PPE probability (p_i)
        ppe_proba = profiling_outputs.get('ppe', {}).get('probabilities', np.array([0.33, 0.34, 0.33]))
        
        # Weight probabilities: High > Medium > Low
        weighted_prob = (ppe_proba[2] * 1.0 +  # High
                        ppe_proba[1] * 0.6 +   # Medium
                        ppe_proba[0] * 0.3)    # Low
        
        # Normalize theta to [0, 1] range (assuming θ ∈ [-3, 3])
        theta_normalized = (theta + 3) / 6
        theta_normalized = np.clip(theta_normalized, 0, 1)
        
        # Compute composite
        lambda1 = 0.5  # Weight for probability
        lambda2 = 0.5  # Weight for ability
        
        alpha = lambda1 * weighted_prob + lambda2 * theta_normalized
        
        return float(alpha)
    
    def _compute_behavioral_stability(self, consistency_index: float) -> float:
        """
        Compute behavioral stability β_i.
        
        Corresponds to line 2 of Algorithm 7: β_i = exp(-κ_i)
        
        Args:
            consistency_index: Standard deviation of performance (κ_i)
            
        Returns:
            Behavioral stability score [0, 1]
        """
        # Normalize consistency index (assuming κ ∈ [0, 2])
        kappa_normalized = np.clip(consistency_index / 2.0, 0, 1)
        
        # Higher consistency (lower κ) → higher stability
        beta = np.exp(-kappa_normalized)
        
        return float(beta)
    
    def _compute_cluster_advantage(self, profiling_outputs: Dict) -> float:
        """
        Compute cluster advantage γ_i.
        
        Corresponds to line 3 of Algorithm 7: γ_i = ClusterWeight(c_i)
        
        Args:
            profiling_outputs: Profiling model outputs
            
        Returns:
            Cluster advantage score
        """
        # Extract cluster ID from LBSM
        cluster_id = profiling_outputs.get('lbsm', {}).get('cluster_id', 1)
        
        # Get advantage from configuration
        gamma = self.cluster_advantages.get(cluster_id, 0.6)
        
        return float(gamma)
    
    def _compute_latent_strength(self, profiling_outputs: Dict) -> float:
        """
        Compute latent strength δ_i.
        
        Corresponds to line 4 of Algorithm 7: δ_i = ||h_i||_2
        
        Args:
            profiling_outputs: Profiling model outputs
            
        Returns:
            Latent strength score
        """
        # Extract embedding norm from LLPE
        embedding_norm = profiling_outputs.get('llpe', {}).get('embedding_norm', 0.0)
        
        # Normalize (assuming typical norms are in [0, 10])
        delta = np.clip(embedding_norm / 10.0, 0, 1)
        
        return float(delta)
    
    def _calculate_benchmark_index(self,
                                   alpha: float,
                                   beta: float,
                                   gamma: float,
                                   delta: float) -> float:
        """
        Calculate benchmark index B_i.
        
        Corresponds to line 5 of Algorithm 7:
        B_i = ω1*α_i + ω2*β_i + ω3*γ_i + ω4*δ_i
        
        Args:
            alpha: Performance composite
            beta: Behavioral stability
            gamma: Cluster advantage
            delta: Latent strength
            
        Returns:
            Benchmark index [0, 1]
        """
        weights = self.benchmark_weights
        
        benchmark_index = (
            weights['performance'] * alpha +
            weights['stability'] * beta +
            weights['cluster'] * gamma +
            weights['latent'] * delta
        )
        
        return float(np.clip(benchmark_index, 0, 1))
    
    def _calculate_scholarship_score(self,
                                    benchmark_index: float,
                                    theta: float,
                                    delta: float) -> float:
        """
        Calculate scholarship suitability score S_i.
        
        Corresponds to line 6 of Algorithm 7:
        S_i = σ(η1*B_i + η2*θ_i + η3*δ_i)
        
        Args:
            benchmark_index: Benchmark index B_i
            theta: IRT ability estimate
            delta: Latent strength
            
        Returns:
            Scholarship score [0, 1]
        """
        weights = self.scholarship_weights
        
        # Normalize theta
        theta_normalized = (theta + 3) / 6
        theta_normalized = np.clip(theta_normalized, 0, 1)
        
        # Linear combination
        score = (
            weights['benchmark'] * benchmark_index +
            weights['ability'] * theta_normalized +
            weights['latent_strength'] * delta
        )
        
        # Apply sigmoid for smooth [0, 1] mapping
        scholarship_score = 1 / (1 + np.exp(-5 * (score - 0.5)))
        
        return float(scholarship_score)
    
    def _classify_performance(self, benchmark_index: float) -> str:
        """
        Classify performance into bands.
        
        Args:
            benchmark_index: Benchmark index
            
        Returns:
            Performance band ('Low', 'Average', 'Good', 'Excellent')
        """
        if benchmark_index < 0.4:
            return 'Low'
        elif benchmark_index < 0.6:
            return 'Average'
        elif benchmark_index < 0.8:
            return 'Good'
        else:
            return 'Excellent'
    
    def batch_compute_scores(self,
                            learner_data: List[Dict]) -> List[BenchmarkScore]:
        """
        Compute scores for multiple learners and add ranking.
        
        Args:
            learner_data: List of dictionaries with learner information.
                         Each dict should contain:
                         - learner_id
                         - profiling_outputs
                         - theta
                         - consistency_index
            
        Returns:
            List of BenchmarkScore objects with ranking
        """
        logger.info(f"Computing benchmark scores for {len(learner_data)} learners")
        
        scores = []
        
        # Compute individual scores
        for data in learner_data:
            score = self.compute_benchmark_score(
                learner_id=data['learner_id'],
                profiling_outputs=data['profiling_outputs'],
                theta=data['theta'],
                consistency_index=data['consistency_index']
            )
            scores.append(score)
        
        # Sort by benchmark index (descending)
        scores.sort(key=lambda x: x.benchmark_index, reverse=True)
        
        # Assign percentile ranks
        n = len(scores)
        for i, score in enumerate(scores):
            # Rank 1 is highest
            rank = i + 1
            percentile = ((n - rank) / n) * 100
            score.rank_percentile = percentile
        
        logger.info(f"Scoring complete. {sum(s.scholarship_eligible for s in scores)} "
                   f"learners eligible for scholarship")
        
        return scores
    
    def get_summary_statistics(self, scores: List[BenchmarkScore]) -> Dict:
        """
        Get summary statistics for a collection of scores.
        
        Args:
            scores: List of BenchmarkScore objects
            
        Returns:
            Dictionary with summary statistics
        """
        benchmark_indices = [s.benchmark_index for s in scores]
        scholarship_scores = [s.scholarship_score for s in scores]
        
        summary = {
            'n_learners': len(scores),
            'benchmark_index': {
                'mean': float(np.mean(benchmark_indices)),
                'std': float(np.std(benchmark_indices)),
                'min': float(np.min(benchmark_indices)),
                'max': float(np.max(benchmark_indices)),
                'median': float(np.median(benchmark_indices))
            },
            'scholarship_score': {
                'mean': float(np.mean(scholarship_scores)),
                'std': float(np.std(scholarship_scores)),
                'min': float(np.min(scholarship_scores)),
                'max': float(np.max(scholarship_scores)),
                'median': float(np.median(scholarship_scores))
            },
            'scholarship_eligible': sum(s.scholarship_eligible for s in scores),
            'scholarship_eligible_pct': float(sum(s.scholarship_eligible for s in scores) / len(scores) * 100),
            'performance_bands': {
                'Low': sum(s.performance_band == 'Low' for s in scores),
                'Average': sum(s.performance_band == 'Average' for s in scores),
                'Good': sum(s.performance_band == 'Good' for s in scores),
                'Excellent': sum(s.performance_band == 'Excellent' for s in scores)
            }
        }
        
        return summary
    
    def export_rankings(self, scores: List[BenchmarkScore], filepath: str):
        """
        Export rankings to CSV file.
        
        Args:
            scores: List of BenchmarkScore objects
            filepath: Output file path
        """
        data = []
        for score in scores:
            data.append({
                'Learner_ID': score.learner_id,
                'Benchmark_Index': score.benchmark_index,
                'Scholarship_Score': score.scholarship_score,
                'Rank_Percentile': score.rank_percentile,
                'Performance_Band': score.performance_band,
                'Scholarship_Eligible': score.scholarship_eligible,
                'Performance_Composite': score.performance_composite,
                'Behavioral_Stability': score.behavioral_stability,
                'Cluster_Advantage': score.cluster_advantage,
                'Latent_Strength': score.latent_strength
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Rankings exported to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("BSSCA (BENCHMARK & SCHOLARSHIP SUITABILITY) DEMO")
    print("=" * 80)
    
    # Generate sample learner data
    np.random.seed(42)
    n_learners = 20
    
    learner_data = []
    for i in range(n_learners):
        # Simulate profiling outputs
        profiling_outputs = {
            'ppe': {
                'probabilities': np.random.dirichlet([1, 1, 1]),
                'prediction': np.random.choice([0, 1, 2]),
                'confidence': np.random.uniform(0.5, 0.95)
            },
            'prem': {
                'prediction': np.random.choice([0, 1, 2])
            },
            'lbsm': {
                'cluster_id': np.random.choice([0, 1, 2]),
                'cluster_name': ['Low', 'Medium', 'High'][np.random.choice([0, 1, 2])]
            },
            'llpe': {
                'embedding': np.random.randn(32),
                'embedding_norm': np.random.uniform(0, 10),
                'probabilities': np.random.dirichlet([1, 1, 1]),
                'prediction': np.random.choice([0, 1, 2])
            }
        }
        
        learner_data.append({
            'learner_id': f'S{str(i+1).zfill(4)}',
            'profiling_outputs': profiling_outputs,
            'theta': np.random.normal(0, 1),
            'consistency_index': np.random.uniform(0, 2)
        })
    
    # Initialize BSSCA
    bssca = BenchmarkScholarshipComputation()
    
    # Compute scores
    print("\n" + "=" * 80)
    print("COMPUTING BENCHMARK SCORES")
    print("=" * 80)
    
    scores = bssca.batch_compute_scores(learner_data)
    
    # Display top 10
    print("\nTop 10 Learners:")
    print(f"{'Rank':<6} {'ID':<8} {'Benchmark':<12} {'Scholarship':<13} {'Eligible':<10} {'Band':<12}")
    print("-" * 75)
    
    for i, score in enumerate(scores[:10], 1):
        print(f"{i:<6} {score.learner_id:<8} {score.benchmark_index:<12.4f} "
              f"{score.scholarship_score:<13.4f} {str(score.scholarship_eligible):<10} "
              f"{score.performance_band:<12}")
    
    # Get summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary = bssca.get_summary_statistics(scores)
    
    print(f"\nTotal learners: {summary['n_learners']}")
    print(f"\nBenchmark Index:")
    print(f"  Mean: {summary['benchmark_index']['mean']:.4f}")
    print(f"  Std: {summary['benchmark_index']['std']:.4f}")
    print(f"  Range: [{summary['benchmark_index']['min']:.4f}, {summary['benchmark_index']['max']:.4f}]")
    
    print(f"\nScholarship Eligibility:")
    print(f"  Eligible: {summary['scholarship_eligible']} ({summary['scholarship_eligible_pct']:.1f}%)")
    
    print(f"\nPerformance Distribution:")
    for band, count in summary['performance_bands'].items():
        print(f"  {band}: {count}")
    
    # Export rankings
    print("\n" + "=" * 80)
    print("EXPORTING RANKINGS")
    print("=" * 80)
    
    bssca.export_rankings(scores, 'benchmark_rankings.csv')
    
    print("\n" + "=" * 80)
    print("BSSCA Demo Complete!")
    print("=" * 80)
