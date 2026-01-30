"""
Adaptive Profiling and Difficulty Estimation Algorithm (APDEA)
Implements Algorithm 5 from the paper

Fuses outputs from PPE, PREM, LBSM, LLPE to produce adaptive learner state:
s_i = {θ_i, B*(i), d_i}
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from config import APDEA_CONFIG, IRT_CONFIG, BLOOM_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearnerState:
    """
    Adaptive learner state containing:
    - theta: IRT ability estimate
    - bloom_level: Recommended Bloom cognitive level
    - difficulty_band: Difficulty assignment ('Easy', 'Medium', 'Hard')
    - confidence: Confidence in the estimate
    """
    theta: float
    bloom_level: int
    bloom_level_name: str
    difficulty_band: str
    confidence: float
    profiling_scores: Dict
    
    def __repr__(self):
        return (f"LearnerState(theta={self.theta:.3f}, "
                f"bloom={self.bloom_level_name}, "
                f"difficulty={self.difficulty_band}, "
                f"confidence={self.confidence:.3f})")


class AdaptiveProfilingDifficultyEstimation:
    """
    APDEA: Adaptive Profiling and Difficulty Estimation Algorithm.
    
    Implements Algorithm 5 from the paper:
    1. Extract profiling outputs (PPE, PREM, LBSM, LLPE)
    2. Estimate ability using IRT 2PL
    3. Recommend Bloom level
    4. Assign difficulty band
    5. Return learner state s_i
    """
    
    def __init__(self, 
                 ppe_model=None,
                 prem_model=None,
                 lbsm_model=None,
                 llpe_model=None,
                 irt_model=None,
                 config: Optional[dict] = None):
        """
        Initialize APDEA with profiling models.
        
        Args:
            ppe_model: Performance Probability Estimator
            prem_model: Pedagogical Rule Extraction Model
            lbsm_model: Learner Behaviour Segmentation Module
            llpe_model: Latent Learning Pattern Extractor
            irt_model: IRT model for ability estimation
            config: Configuration dictionary
        """
        if config is None:
            config = APDEA_CONFIG
        
        self.config = config
        self.ppe_model = ppe_model
        self.prem_model = prem_model
        self.lbsm_model = lbsm_model
        self.llpe_model = llpe_model
        self.irt_model = irt_model
        
        # Fusion weights
        self.fusion_weights = config['fusion_weights']
        
        logger.info("APDEA initialized")
        logger.info(f"  Fusion weights: {self.fusion_weights}")
    
    def compute_learner_state(self, 
                             learner_features: np.ndarray,
                             learner_responses: Optional[np.ndarray] = None) -> LearnerState:
        """
        Compute adaptive learner state by fusing all profiling outputs.
        
        Implements Algorithm 5 (APDEA).
        
        Args:
            learner_features: Feature vector for learner (d,)
            learner_responses: Binary response vector for IRT (N_items,)
            
        Returns:
            LearnerState object with theta, Bloom level, and difficulty band
        """
        logger.debug("Computing learner state...")
        
        if learner_features.ndim == 1:
            learner_features = learner_features.reshape(1, -1)
        
        # Step 1: Extract profiling outputs
        profiling_outputs = self._extract_profiling_outputs(learner_features)
        
        # Step 2: Estimate ability using IRT
        theta = self._estimate_ability(learner_responses, profiling_outputs)
        
        # Step 3: Recommend Bloom cognitive level
        bloom_level = self._recommend_bloom_level(profiling_outputs, theta)
        
        # Step 4: Assign difficulty band
        difficulty_band = self._assign_difficulty_band(theta)
        
        # Step 5: Compute confidence
        confidence = self._compute_confidence(profiling_outputs)
        
        # Create learner state
        bloom_names = {1: 'Remember', 2: 'Apply', 3: 'Analyze'}
        learner_state = LearnerState(
            theta=float(theta),
            bloom_level=bloom_level,
            bloom_level_name=bloom_names[bloom_level],
            difficulty_band=difficulty_band,
            confidence=float(confidence),
            profiling_scores=profiling_outputs
        )
        
        logger.debug(f"Learner state computed: {learner_state}")
        
        return learner_state
    
    def _extract_profiling_outputs(self, learner_features: np.ndarray) -> Dict:
        """
        Extract outputs from all profiling models.
        
        Corresponds to lines 1-4 of Algorithm 5.
        
        Args:
            learner_features: Feature vector
            
        Returns:
            Dictionary with profiling outputs
        """
        outputs = {}
        
        # PPE: Performance probability
        if self.ppe_model is not None and self.ppe_model.is_trained:
            ppe_proba = self.ppe_model.predict_proba(learner_features)[0]
            outputs['ppe'] = {
                'probabilities': ppe_proba,
                'prediction': np.argmax(ppe_proba),
                'confidence': np.max(ppe_proba)
            }
        else:
            # Default fallback
            outputs['ppe'] = {
                'probabilities': np.array([0.33, 0.34, 0.33]),
                'prediction': 1,
                'confidence': 0.34
            }
        
        # PREM: Rule-based category
        if self.prem_model is not None and self.prem_model.is_trained:
            prem_pred = self.prem_model.predict(learner_features)[0]
            outputs['prem'] = {
                'prediction': prem_pred,
                'rules': 'See model for rules'
            }
        else:
            outputs['prem'] = {'prediction': 1}
        
        # LBSM: Behavioral cluster
        if self.lbsm_model is not None and self.lbsm_model.is_trained:
            cluster_id = self.lbsm_model.predict(learner_features)[0]
            outputs['lbsm'] = {
                'cluster_id': cluster_id,
                'cluster_name': ['Low', 'Medium', 'High'][cluster_id]
            }
        else:
            outputs['lbsm'] = {'cluster_id': 1, 'cluster_name': 'Medium'}
        
        # LLPE: Latent embedding
        if self.llpe_model is not None and self.llpe_model.is_trained:
            embedding = self.llpe_model.get_embedding(learner_features)[0]
            llpe_proba = self.llpe_model.predict_proba(learner_features)[0]
            outputs['llpe'] = {
                'embedding': embedding,
                'embedding_norm': np.linalg.norm(embedding),
                'probabilities': llpe_proba,
                'prediction': np.argmax(llpe_proba)
            }
        else:
            outputs['llpe'] = {
                'embedding': np.zeros(32),
                'embedding_norm': 0.0,
                'probabilities': np.array([0.33, 0.34, 0.33]),
                'prediction': 1
            }
        
        return outputs
    
    def _estimate_ability(self, 
                         learner_responses: Optional[np.ndarray],
                         profiling_outputs: Dict) -> float:
        """
        Estimate learner ability θ using IRT 2PL.
        
        Corresponds to lines 5-7 of Algorithm 5.
        
        Args:
            learner_responses: Binary response vector
            profiling_outputs: Profiling model outputs
            
        Returns:
            Ability estimate θ
        """
        # If IRT model available and responses provided, use IRT
        if (self.irt_model is not None and 
            self.irt_model.is_calibrated and 
            learner_responses is not None):
            theta = self.irt_model.estimate_ability(learner_responses)
            return float(theta)
        
        # Otherwise, approximate from profiling outputs
        # Map performance class to ability
        ppe_pred = profiling_outputs['ppe']['prediction']
        
        # Map: 0 (Low) → -1.0, 1 (Medium) → 0.0, 2 (High) → 1.0
        ability_map = {0: -1.0, 1: 0.0, 2: 1.0}
        theta_approx = ability_map.get(ppe_pred, 0.0)
        
        # Adjust based on confidence
        confidence = profiling_outputs['ppe']['confidence']
        if confidence < 0.5:
            # Low confidence, move toward mean
            theta_approx *= confidence * 2
        
        return theta_approx
    
    def _recommend_bloom_level(self, 
                               profiling_outputs: Dict,
                               theta: float) -> int:
        """
        Recommend Bloom cognitive level based on profiling and ability.
        
        Corresponds to lines 8-9 of Algorithm 5:
        B*(i) = argmax_b P(b | p_i, r_i, c_i, h_i)
        
        Args:
            profiling_outputs: Profiling model outputs
            theta: Ability estimate
            
        Returns:
            Bloom level (1: Remember, 2: Apply, 3: Analyze)
        """
        method = self.config['bloom_prediction_method']
        
        if method == 'weighted_voting':
            # Weighted voting from profiling models
            bloom_votes = np.zeros(3)  # [Remember, Apply, Analyze]
            
            # PPE contribution
            ppe_pred = profiling_outputs['ppe']['prediction']
            bloom_votes[ppe_pred] += self.fusion_weights['probability']
            
            # PREM contribution
            prem_pred = profiling_outputs['prem']['prediction']
            bloom_votes[prem_pred] += self.fusion_weights['rules']
            
            # LBSM contribution
            lbsm_cluster = profiling_outputs['lbsm']['cluster_id']
            bloom_votes[lbsm_cluster] += self.fusion_weights['clusters']
            
            # LLPE contribution
            llpe_pred = profiling_outputs['llpe']['prediction']
            bloom_votes[llpe_pred] += self.fusion_weights['latent']
            
            # Select level with highest vote
            bloom_level = int(np.argmax(bloom_votes)) + 1  # 1-indexed
        
        elif method == 'ability_based':
            # Map ability to Bloom level
            if theta < -0.5:
                bloom_level = 1  # Remember
            elif theta < 0.5:
                bloom_level = 2  # Apply
            else:
                bloom_level = 3  # Analyze
        
        else:
            # Default: use PPE prediction
            bloom_level = profiling_outputs['ppe']['prediction'] + 1
        
        # Ensure level is in valid range
        bloom_level = np.clip(bloom_level, 1, 3)
        
        return int(bloom_level)
    
    def _assign_difficulty_band(self, theta: float) -> str:
        """
        Assign difficulty band based on ability.
        
        Corresponds to lines 10-15 of Algorithm 5.
        
        Args:
            theta: Ability estimate
            
        Returns:
            Difficulty band ('Easy', 'Medium', 'Hard')
        """
        thresholds = IRT_CONFIG['ability_thresholds']
        
        if theta < thresholds['easy']:
            return 'Easy'
        elif theta < thresholds['medium']:
            return 'Medium'
        else:
            return 'Hard'
    
    def _compute_confidence(self, profiling_outputs: Dict) -> float:
        """
        Compute overall confidence in the learner state estimate.
        
        Args:
            profiling_outputs: Profiling model outputs
            
        Returns:
            Confidence score [0, 1]
        """
        # Average confidence from probabilistic models
        ppe_conf = profiling_outputs['ppe']['confidence']
        llpe_conf = profiling_outputs['llpe']['probabilities'].max()
        
        # Check agreement between models
        predictions = [
            profiling_outputs['ppe']['prediction'],
            profiling_outputs['prem']['prediction'],
            profiling_outputs['lbsm']['cluster_id'],
            profiling_outputs['llpe']['prediction']
        ]
        
        # Agreement score: fraction of models that agree with majority
        majority_pred = max(set(predictions), key=predictions.count)
        agreement = sum(p == majority_pred for p in predictions) / len(predictions)
        
        # Combined confidence
        confidence = 0.5 * (ppe_conf + llpe_conf) * agreement
        
        return float(np.clip(confidence, 0, 1))
    
    def batch_compute_learner_states(self,
                                    learner_features: np.ndarray,
                                    learner_responses: Optional[np.ndarray] = None) -> list:
        """
        Compute learner states for multiple learners.
        
        Args:
            learner_features: Feature matrix (N x d)
            learner_responses: Response matrix (N x N_items), optional
            
        Returns:
            List of LearnerState objects
        """
        n_learners = learner_features.shape[0]
        states = []
        
        logger.info(f"Computing states for {n_learners} learners...")
        
        for i in range(n_learners):
            features = learner_features[i]
            responses = learner_responses[i] if learner_responses is not None else None
            
            state = self.compute_learner_state(features, responses)
            states.append(state)
        
        logger.info(f"Computed {len(states)} learner states")
        
        return states
    
    def get_state_summary(self, states: list) -> Dict:
        """
        Get summary statistics for a collection of learner states.
        
        Args:
            states: List of LearnerState objects
            
        Returns:
            Dictionary with summary statistics
        """
        thetas = [s.theta for s in states]
        bloom_levels = [s.bloom_level for s in states]
        difficulties = [s.difficulty_band for s in states]
        confidences = [s.confidence for s in states]
        
        summary = {
            'n_learners': len(states),
            'ability': {
                'mean': np.mean(thetas),
                'std': np.std(thetas),
                'min': np.min(thetas),
                'max': np.max(thetas)
            },
            'bloom_distribution': {
                'Remember': bloom_levels.count(1),
                'Apply': bloom_levels.count(2),
                'Analyze': bloom_levels.count(3)
            },
            'difficulty_distribution': {
                'Easy': difficulties.count('Easy'),
                'Medium': difficulties.count('Medium'),
                'Hard': difficulties.count('Hard')
            },
            'confidence': {
                'mean': np.mean(confidences),
                'std': np.std(confidences)
            }
        }
        
        return summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("APDEA (ADAPTIVE PROFILING & DIFFICULTY ESTIMATION) DEMO")
    print("=" * 80)
    
    # Simulate profiling models with dummy data
    class DummyModel:
        def __init__(self):
            self.is_trained = True
            self.is_calibrated = True
        
        def predict(self, X):
            return np.random.choice([0, 1, 2], size=len(X) if X.ndim > 1 else 1)
        
        def predict_proba(self, X):
            n = len(X) if X.ndim > 1 else 1
            probs = np.random.dirichlet([1, 1, 1], size=n)
            return probs
        
        def get_embedding(self, X):
            n = len(X) if X.ndim > 1 else 1
            return np.random.randn(n, 32)
        
        def estimate_ability(self, responses):
            return np.random.normal(0, 1)
    
    # Initialize APDEA with dummy models
    apdea = AdaptiveProfilingDifficultyEstimation(
        ppe_model=DummyModel(),
        prem_model=DummyModel(),
        lbsm_model=DummyModel(),
        llpe_model=DummyModel(),
        irt_model=DummyModel()
    )
    
    # Generate sample data
    np.random.seed(42)
    n_learners = 10
    n_features = 20
    n_items = 12
    
    learner_features = np.random.randn(n_learners, n_features)
    learner_responses = (np.random.random((n_learners, n_items)) > 0.5).astype(int)
    
    # Compute states
    print("\n" + "=" * 80)
    print("COMPUTING LEARNER STATES")
    print("=" * 80)
    
    states = apdea.batch_compute_learner_states(learner_features, learner_responses)
    
    # Display first few states
    print("\nFirst 5 Learner States:")
    for i, state in enumerate(states[:5], 1):
        print(f"\nLearner {i}:")
        print(f"  {state}")
        print(f"  Profiling confidence: {state.confidence:.3f}")
    
    # Get summary
    print("\n" + "=" * 80)
    print("STATE SUMMARY")
    print("=" * 80)
    
    summary = apdea.get_state_summary(states)
    
    print(f"\nTotal learners: {summary['n_learners']}")
    print(f"\nAbility Statistics:")
    print(f"  Mean: {summary['ability']['mean']:.3f}")
    print(f"  Std: {summary['ability']['std']:.3f}")
    print(f"  Range: [{summary['ability']['min']:.3f}, {summary['ability']['max']:.3f}]")
    
    print(f"\nBloom Level Distribution:")
    for level, count in summary['bloom_distribution'].items():
        print(f"  {level}: {count}")
    
    print(f"\nDifficulty Distribution:")
    for diff, count in summary['difficulty_distribution'].items():
        print(f"  {diff}: {count}")
    
    print("\n" + "=" * 80)
    print("APDEA Demo Complete!")
    print("=" * 80)
