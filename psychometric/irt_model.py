"""
Item Response Theory (IRT) Model
Two-Parameter Logistic (2PL) Model for ability estimation
Equation: P(correct | θ, a, b) = 1 / (1 + exp(-a(θ - b)))
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, List
from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.stats import norm
import joblib
from pathlib import Path

from config import IRT_CONFIG, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoParameterLogisticModel:
    """
    Two-Parameter Logistic (2PL) IRT Model.
    
    P(correct | θ, a, b) = 1 / (1 + exp(-a(θ - b)))
    
    where:
    - θ (theta): learner ability
    - a: item discrimination (slope)
    - b: item difficulty (location)
    """
    
    @staticmethod
    def probability(theta: float, a: float, b: float) -> float:
        """
        Compute probability of correct response.
        
        Args:
            theta: Learner ability
            a: Item discrimination
            b: Item difficulty
            
        Returns:
            Probability of correct response [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-a * (theta - b)))
    
    @staticmethod
    def log_likelihood(params: np.ndarray, responses: np.ndarray, 
                      item_params: Optional[np.ndarray] = None) -> float:
        """
        Compute log-likelihood for parameter estimation.
        
        Args:
            params: Parameters to estimate
            responses: Binary response matrix (N_learners x N_items)
            item_params: Fixed item parameters if estimating theta only
            
        Returns:
            Negative log-likelihood (for minimization)
        """
        if item_params is not None:
            # Estimating theta with fixed item params
            theta = params
            probs = TwoParameterLogisticModel.probability(
                theta, item_params[:, 0], item_params[:, 1]
            )
        else:
            # Estimating item parameters with fixed theta
            n_items = responses.shape[1]
            a_params = params[:n_items]
            b_params = params[n_items:]
            theta = 0.0  # Assume standard ability for item calibration
            probs = TwoParameterLogisticModel.probability(theta, a_params, b_params)
        
        # Compute log-likelihood
        ll = np.sum(
            responses * np.log(probs + 1e-10) + 
            (1 - responses) * np.log(1 - probs + 1e-10)
        )
        
        return -ll  # Return negative for minimization
    
    @staticmethod
    def information(theta: float, a: float, b: float) -> float:
        """
        Compute Fisher information at given ability level.
        
        Args:
            theta: Learner ability
            a: Item discrimination
            b: Item difficulty
            
        Returns:
            Information value
        """
        p = TwoParameterLogisticModel.probability(theta, a, b)
        return a**2 * p * (1 - p)


class IRTEstimator:
    """
    IRT Estimator for ability and item parameter estimation.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize IRT estimator.
        
        Args:
            config: Configuration dictionary. Uses IRT_CONFIG if None.
        """
        if config is None:
            config = IRT_CONFIG
        
        self.config = config
        self.item_params = None  # (a, b) for each item
        self.learner_abilities = None  # θ for each learner
        self.is_calibrated = False
        
        logger.info("IRT Estimator initialized with 2PL model")
        logger.info(f"  Ability range: {config['ability_range']}")
        logger.info(f"  Difficulty range: {config['difficulty_range']}")
        logger.info(f"  Discrimination range: {config['discrimination_range']}")
    
    def calibrate_items(self, responses: np.ndarray, 
                       initial_abilities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calibrate item parameters (a, b) given response data.
        
        Uses Marginal Maximum Likelihood Estimation (MMLE).
        
        Args:
            responses: Binary response matrix (N_learners x N_items)
            initial_abilities: Initial ability estimates (optional)
            
        Returns:
            Item parameters array (N_items x 2) with columns [a, b]
        """
        logger.info("Calibrating item parameters...")
        
        n_learners, n_items = responses.shape
        
        if initial_abilities is None:
            # Initialize with proportion correct
            initial_abilities = norm.ppf(
                np.clip(responses.mean(axis=1), 0.01, 0.99)
            )
        
        # Initialize item parameters
        item_params = np.zeros((n_items, 2))
        
        for item_idx in range(n_items):
            item_responses = responses[:, item_idx]
            
            # Initial values
            p_correct = item_responses.mean()
            b_init = -norm.ppf(p_correct) if 0 < p_correct < 1 else 0.0
            a_init = 1.0
            
            # Bounds
            bounds = [
                self.config['discrimination_range'],
                self.config['difficulty_range']
            ]
            
            # Objective function for this item
            def objective(params):
                a, b = params
                probs = TwoParameterLogisticModel.probability(
                    initial_abilities, a, b
                )
                ll = np.sum(
                    item_responses * np.log(probs + 1e-10) +
                    (1 - item_responses) * np.log(1 - probs + 1e-10)
                )
                return -ll
            
            # Optimize
            result = minimize(
                objective,
                x0=[a_init, b_init],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            item_params[item_idx] = result.x
        
        self.item_params = item_params
        self.is_calibrated = True
        
        logger.info(f"Item calibration complete")
        logger.info(f"  Mean discrimination (a): {item_params[:, 0].mean():.3f}")
        logger.info(f"  Mean difficulty (b): {item_params[:, 1].mean():.3f}")
        
        return item_params
    
    def estimate_ability(self, responses: np.ndarray, 
                        item_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate learner abilities given responses and item parameters.
        
        Uses Maximum Likelihood Estimation (MLE) or Expected A Posteriori (EAP).
        
        Args:
            responses: Binary response matrix (N_learners x N_items) or vector (N_items,)
            item_params: Item parameters (N_items x 2). Uses self.item_params if None.
            
        Returns:
            Ability estimates (N_learners,) or scalar
        """
        if item_params is None:
            if self.item_params is None:
                raise ValueError("Item parameters must be calibrated first")
            item_params = self.item_params
        
        # Handle single learner
        single_learner = False
        if responses.ndim == 1:
            responses = responses.reshape(1, -1)
            single_learner = True
        
        n_learners = responses.shape[0]
        abilities = np.zeros(n_learners)
        
        for learner_idx in range(n_learners):
            learner_responses = responses[learner_idx]
            
            # Objective function for this learner
            def objective(theta):
                probs = TwoParameterLogisticModel.probability(
                    theta[0], item_params[:, 0], item_params[:, 1]
                )
                ll = np.sum(
                    learner_responses * np.log(probs + 1e-10) +
                    (1 - learner_responses) * np.log(1 - probs + 1e-10)
                )
                # Add prior (normal distribution)
                prior = -0.5 * theta[0]**2
                return -(ll + prior)
            
            # Optimize
            result = minimize(
                objective,
                x0=[self.config['initial_ability']],
                bounds=[self.config['ability_range']],
                method='L-BFGS-B'
            )
            
            abilities[learner_idx] = result.x[0]
        
        if single_learner:
            return float(abilities[0])
        
        return abilities
    
    def estimate_ability_eap(self, responses: np.ndarray,
                            item_params: Optional[np.ndarray] = None,
                            prior_mean: float = 0.0,
                            prior_std: float = 1.0,
                            n_quadrature: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate ability using Expected A Posteriori (EAP) method.
        
        Args:
            responses: Binary response vector (N_items,)
            item_params: Item parameters (N_items x 2)
            prior_mean: Prior distribution mean
            prior_std: Prior distribution standard deviation
            n_quadrature: Number of quadrature points
            
        Returns:
            Tuple of (ability_estimate, standard_error)
        """
        if item_params is None:
            item_params = self.item_params
        
        # Quadrature points and weights
        theta_range = np.linspace(
            self.config['ability_range'][0],
            self.config['ability_range'][1],
            n_quadrature
        )
        
        # Prior probabilities
        prior = norm.pdf(theta_range, prior_mean, prior_std)
        
        # Likelihood at each quadrature point
        likelihood = np.ones(n_quadrature)
        for theta_idx, theta in enumerate(theta_range):
            probs = TwoParameterLogisticModel.probability(
                theta, item_params[:, 0], item_params[:, 1]
            )
            likelihood[theta_idx] = np.prod(
                probs ** responses * (1 - probs) ** (1 - responses)
            )
        
        # Posterior
        posterior = likelihood * prior
        posterior /= posterior.sum()
        
        # EAP estimate
        ability_eap = np.sum(theta_range * posterior)
        
        # Standard error
        variance = np.sum((theta_range - ability_eap)**2 * posterior)
        std_error = np.sqrt(variance)
        
        return ability_eap, std_error
    
    def assign_difficulty_band(self, theta: float) -> str:
        """
        Assign difficulty band based on ability estimate.
        
        Args:
            theta: Ability estimate
            
        Returns:
            Difficulty band ('Easy', 'Medium', 'Hard')
        """
        thresholds = self.config['ability_thresholds']
        
        if theta < thresholds['easy']:
            return 'Easy'
        elif theta < thresholds['medium']:
            return 'Medium'
        else:
            return 'Hard'
    
    def get_item_information_curve(self, item_idx: int, 
                                   theta_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get item information curve.
        
        Args:
            item_idx: Index of item
            theta_range: Range of abilities to plot (optional)
            
        Returns:
            Tuple of (theta_values, information_values)
        """
        if self.item_params is None:
            raise ValueError("Item parameters not calibrated")
        
        if theta_range is None:
            theta_range = np.linspace(
                self.config['ability_range'][0],
                self.config['ability_range'][1],
                100
            )
        
        a, b = self.item_params[item_idx]
        information = TwoParameterLogisticModel.information(theta_range, a, b)
        
        return theta_range, information
    
    def get_test_information(self, theta_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test information curve (sum of all item information).
        
        Args:
            theta_range: Range of abilities to plot (optional)
            
        Returns:
            Tuple of (theta_values, total_information)
        """
        if self.item_params is None:
            raise ValueError("Item parameters not calibrated")
        
        if theta_range is None:
            theta_range = np.linspace(
                self.config['ability_range'][0],
                self.config['ability_range'][0],
                100
            )
        
        total_info = np.zeros_like(theta_range)
        
        for item_idx in range(len(self.item_params)):
            a, b = self.item_params[item_idx]
            total_info += TwoParameterLogisticModel.information(theta_range, a, b)
        
        return theta_range, total_info
    
    def predict_response_probability(self, theta: float, item_idx: int) -> float:
        """
        Predict probability of correct response.
        
        Args:
            theta: Learner ability
            item_idx: Index of item
            
        Returns:
            Probability of correct response
        """
        if self.item_params is None:
            raise ValueError("Item parameters not calibrated")
        
        a, b = self.item_params[item_idx]
        return TwoParameterLogisticModel.probability(theta, a, b)
    
    def get_learner_profile(self, responses: np.ndarray) -> dict:
        """
        Get comprehensive IRT-based learner profile.
        
        Args:
            responses: Binary response vector (N_items,)
            
        Returns:
            Dictionary with ability estimates and characteristics
        """
        # MLE ability
        theta_mle = self.estimate_ability(responses)
        
        # EAP ability with standard error
        theta_eap, se = self.estimate_ability_eap(responses)
        
        # Difficulty band
        difficulty_band = self.assign_difficulty_band(theta_mle)
        
        # Expected scores
        expected_probs = np.array([
            self.predict_response_probability(theta_mle, i)
            for i in range(len(responses))
        ])
        expected_score = expected_probs.sum()
        actual_score = responses.sum()
        
        profile = {
            'ability_mle': float(theta_mle),
            'ability_eap': float(theta_eap),
            'ability_se': float(se),
            'confidence_interval_95': [
                float(theta_eap - 1.96 * se),
                float(theta_eap + 1.96 * se)
            ],
            'difficulty_band': difficulty_band,
            'actual_score': int(actual_score),
            'expected_score': float(expected_score),
            'score_discrepancy': float(actual_score - expected_score)
        }
        
        return profile
    
    def save_model(self, filepath: Optional[str] = None):
        """Save calibrated model"""
        if not self.is_calibrated:
            raise ValueError("No calibrated model to save")
        
        if filepath is None:
            filepath = MODELS_DIR / 'irt_model.pkl'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'config': self.config,
            'item_params': self.item_params,
            'is_calibrated': self.is_calibrated
        }, filepath)
        
        logger.info(f"IRT model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None):
        """Load calibrated model"""
        if filepath is None:
            filepath = MODELS_DIR / 'irt_model.pkl'
        
        saved_data = joblib.load(filepath)
        
        self.config = saved_data['config']
        self.item_params = saved_data['item_params']
        self.is_calibrated = saved_data['is_calibrated']
        
        logger.info(f"IRT model loaded from {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("IRT (ITEM RESPONSE THEORY) MODEL DEMO")
    print("=" * 80)
    
    # Generate synthetic response data
    np.random.seed(42)
    
    n_learners = 200
    n_items = 12
    
    # True parameters
    true_abilities = np.random.normal(0, 1, n_learners)
    true_a = np.random.uniform(0.5, 2.5, n_items)
    true_b = np.random.uniform(-2, 2, n_items)
    
    # Generate responses
    responses = np.zeros((n_learners, n_items))
    for i in range(n_learners):
        for j in range(n_items):
            prob = TwoParameterLogisticModel.probability(
                true_abilities[i], true_a[j], true_b[j]
            )
            responses[i, j] = np.random.random() < prob
    
    print(f"\nGenerated {n_learners} learners × {n_items} items")
    print(f"Response matrix shape: {responses.shape}")
    print(f"Mean score: {responses.mean():.3f}")
    
    # Initialize IRT estimator
    irt = IRTEstimator()
    
    # Calibrate items
    print("\n" + "=" * 80)
    print("CALIBRATING ITEM PARAMETERS")
    print("=" * 80)
    
    item_params = irt.calibrate_items(responses)
    
    print("\nEstimated Item Parameters:")
    print("Item   |   a (disc)  |   b (diff)")
    print("-" * 40)
    for i, (a, b) in enumerate(item_params):
        print(f"  {i+1:2d}   |   {a:6.3f}    |   {b:6.3f}")
    
    # Estimate abilities
    print("\n" + "=" * 80)
    print("ESTIMATING LEARNER ABILITIES")
    print("=" * 80)
    
    estimated_abilities = irt.estimate_ability(responses)
    
    print(f"\nAbility estimates (first 10 learners):")
    for i in range(min(10, n_learners)):
        print(f"  Learner {i+1}: θ = {estimated_abilities[i]:.3f} "
              f"(difficulty band: {irt.assign_difficulty_band(estimated_abilities[i])})")
    
    # Get learner profile
    print("\n" + "=" * 80)
    print("DETAILED LEARNER PROFILE")
    print("=" * 80)
    
    profile = irt.get_learner_profile(responses[0])
    print("\nLearner 1 Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    # Compare true vs estimated
    print("\n" + "=" * 80)
    print("CALIBRATION ACCURACY")
    print("=" * 80)
    
    ability_correlation = np.corrcoef(true_abilities, estimated_abilities)[0, 1]
    print(f"\nCorrelation (true vs estimated abilities): {ability_correlation:.3f}")
    
    param_correlation_a = np.corrcoef(true_a, item_params[:, 0])[0, 1]
    param_correlation_b = np.corrcoef(true_b, item_params[:, 1])[0, 1]
    print(f"Correlation (true vs estimated discrimination): {param_correlation_a:.3f}")
    print(f"Correlation (true vs estimated difficulty): {param_correlation_b:.3f}")
    
    # Save model
    irt.save_model()
    
    print("\n" + "=" * 80)
    print("IRT Demo Complete!")
    print("=" * 80)
