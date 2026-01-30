"""
Performance Probability Estimator (PPE)
Logistic Regression-based probabilistic performance prediction
Implements Algorithm 1 from the paper
"""

import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib
from pathlib import Path

from config import PPE_CONFIG, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceProbabilityEstimator:
    """
    Performance Probability Estimator (PPE) using Logistic Regression.
    
    Provides probability-based predictions for learner performance categories.
    Equation from paper: P(Y=1|X) = 1 / (1 + exp(-β^T X))
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize PPE model.
        
        Args:
            config: Configuration dictionary. Uses PPE_CONFIG if None.
        """
        if config is None:
            config = PPE_CONFIG
        
        self.config = config
        self.model = None
        self.is_trained = False
        self.classes_ = None
        self.feature_dim = None
        
        # Initialize model based on config
        self._initialize_model()
        
        logger.info("PPE initialized with configuration:")
        logger.info(f"  Learning rate: {config.get('learning_rate', 'default')}")
        logger.info(f"  Max iterations: {config['max_iter']}")
        logger.info(f"  Penalty: {config['penalty']}")
    
    def _initialize_model(self):
        """Initialize the logistic regression model"""
        self.model = LogisticRegression(
            penalty=self.config['penalty'],
            C=self.config['C'],
            solver=self.config['solver'],
            max_iter=self.config['max_iter'],
            random_state=42,
            multi_class='multinomial',  # For multi-class classification
            verbose=0
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train the PPE model.
        Implements Algorithm 1 from the paper.
        
        Args:
            X_train: Training feature matrix (N x d)
            y_train: Training labels (N,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("=" * 80)
        logger.info("TRAINING PERFORMANCE PROBABILITY ESTIMATOR (PPE)")
        logger.info("=" * 80)
        
        self.feature_dim = X_train.shape[1]
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Feature dimension: {self.feature_dim}")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        
        # Train model (implements mini-batch gradient descent internally)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_logloss = log_loss(y_train, y_train_proba)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'train_logloss': train_logloss,
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Training log-loss: {train_logloss:.4f}")
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            y_val_proba = self.model.predict_proba(X_val)
            
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_logloss = log_loss(y_val, y_val_proba)
            
            metrics['val_accuracy'] = val_accuracy
            metrics['val_logloss'] = val_logloss
            
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
            logger.info(f"Validation log-loss: {val_logloss:.4f}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_train, y_train_pred, 
                                                 target_names=['Low', 'Medium', 'High']))
        
        logger.info("=" * 80)
        logger.info("PPE TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Implements: P(Y=c|X) = exp(β_c^T X) / Σ_k exp(β_k^T X)
        
        Args:
            X: Feature matrix (N x d)
            
        Returns:
            Probability matrix (N x C) where C is number of classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix (N x d)
            
        Returns:
            Predicted labels (N,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_performance_probability(self, X: np.ndarray, 
                                   class_label: Optional[int] = None) -> np.ndarray:
        """
        Get probability estimates for specific class or all classes.
        
        This is the core PPE output: P(Performance=class|features)
        
        Args:
            X: Feature matrix (N x d)
            class_label: Specific class to get probability for (None for all)
            
        Returns:
            Probability array
        """
        proba = self.predict_proba(X)
        
        if class_label is not None:
            # Return probability for specific class
            class_idx = np.where(self.classes_ == class_label)[0][0]
            return proba[:, class_idx]
        
        return proba
    
    def get_learner_profile(self, X: np.ndarray) -> dict:
        """
        Get comprehensive learner profile from PPE.
        
        Args:
            X: Feature vector for single learner (d,) or (1, d)
            
        Returns:
            Dictionary with probability estimates and predictions
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        proba = self.predict_proba(X)[0]
        pred = self.predict(X)[0]
        
        profile = {
            'predicted_class': int(pred),
            'predicted_class_name': ['Low', 'Medium', 'High'][pred],
            'class_probabilities': {
                'Low': float(proba[0]),
                'Medium': float(proba[1]),
                'High': float(proba[2])
            },
            'confidence': float(np.max(proba)),
            'uncertainty': float(1 - np.max(proba))
        }
        
        return profile
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> dict:
        """
        Get feature importance based on coefficient magnitudes.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping features to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Average absolute coefficients across classes
        coef_importance = np.abs(self.model.coef_).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.feature_dim)]
        
        importance_dict = dict(zip(feature_names, coef_importance))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
        
        return importance_dict
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model. Uses default if None.
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = MODELS_DIR / 'ppe_model.pkl'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'classes': self.classes_,
            'feature_dim': self.feature_dim
        }, filepath)
        
        logger.info(f"PPE model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        if filepath is None:
            filepath = MODELS_DIR / 'ppe_model.pkl'
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        saved_data = joblib.load(filepath)
        
        self.model = saved_data['model']
        self.config = saved_data['config']
        self.classes_ = saved_data['classes']
        self.feature_dim = saved_data['feature_dim']
        self.is_trained = True
        
        logger.info(f"PPE model loaded from {filepath}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'log_loss': log_loss(y, y_proba),
        }
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, average=None
        )
        
        for i, class_name in enumerate(['Low', 'Medium', 'High']):
            metrics[f'{class_name}_precision'] = precision[i]
            metrics[f'{class_name}_recall'] = recall[i]
            metrics[f'{class_name}_f1'] = f1[i]
        
        # Macro averages
        metrics['macro_precision'] = precision.mean()
        metrics['macro_recall'] = recall.mean()
        metrics['macro_f1'] = f1.mean()
        
        return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PPE (PERFORMANCE PROBABILITY ESTIMATOR) DEMO")
    print("=" * 80)
    
    # Generate synthetic data for demo
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    # Create synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels (0: Low, 1: Medium, 2: High)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.50, 0.25])
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize and train PPE
    ppe = PerformanceProbabilityEstimator()
    
    metrics = ppe.train(X_train, y_train, X_test, y_test)
    
    # Make predictions
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    
    sample_X = X_test[:5]
    probabilities = ppe.predict_proba(sample_X)
    predictions = ppe.predict(sample_X)
    
    print("\nSample predictions:")
    for i in range(5):
        print(f"\nLearner {i+1}:")
        print(f"  Predicted: {['Low', 'Medium', 'High'][predictions[i]]}")
        print(f"  Probabilities: Low={probabilities[i][0]:.3f}, "
              f"Medium={probabilities[i][1]:.3f}, High={probabilities[i][2]:.3f}")
    
    # Get learner profile
    print("\n" + "=" * 80)
    print("LEARNER PROFILE EXAMPLE")
    print("=" * 80)
    
    profile = ppe.get_learner_profile(X_test[0])
    print("\nLearner Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    test_metrics = ppe.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    # Save model
    ppe.save_model()
    
    print("\n" + "=" * 80)
    print("PPE Demo Complete!")
    print("=" * 80)
