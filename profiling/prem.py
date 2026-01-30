"""
Pedagogical Rule Extraction Model (PREM)
Decision Tree-based interpretable rule learning
Implements Algorithm 2 from the paper
"""

import numpy as np
import logging
from typing import Optional, List, Dict
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

from config import PREM_CONFIG, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PedagogicalRuleExtractionModel:
    """
    Pedagogical Rule Extraction Model (PREM) using Decision Trees.
    
    Learns interpretable if-then rules for performance classification.
    Implements recursive tree building with information gain (Algorithm 2).
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize PREM model.
        
        Args:
            config: Configuration dictionary. Uses PREM_CONFIG if None.
        """
        if config is None:
            config = PREM_CONFIG
        
        self.config = config
        self.model = None
        self.is_trained = False
        self.classes_ = None
        self.feature_names = None
        self.rules = []
        
        # Initialize model
        self._initialize_model()
        
        logger.info("PREM initialized with configuration:")
        logger.info(f"  Max depth: {config['max_depth']}")
        logger.info(f"  Min samples split: {config['min_samples_split']}")
        logger.info(f"  Criterion: {config['criterion']}")
    
    def _initialize_model(self):
        """Initialize the decision tree classifier"""
        self.model = DecisionTreeClassifier(
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            criterion=self.config['criterion'],
            max_features=self.config['max_features'],
            class_weight=self.config['class_weight'],
            random_state=42
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None,
             feature_names: Optional[List[str]] = None) -> dict:
        """
        Train the PREM model.
        Implements Algorithm 2: BuildTree recursively via sklearn.
        
        Args:
            X_train: Training feature matrix (N x d)
            y_train: Training labels (N,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features for rule extraction
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("=" * 80)
        logger.info("TRAINING PEDAGOGICAL RULE EXTRACTION MODEL (PREM)")
        logger.info("=" * 80)
        
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"X{i}" for i in range(X_train.shape[1])]
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Feature dimension: {X_train.shape[1]}")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        
        # Train decision tree (implements recursive BuildTree)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # Extract rules
        self.rules = self._extract_rules()
        
        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'tree_depth': self.model.get_depth(),
            'num_leaves': self.model.get_n_leaves(),
            'num_rules': len(self.rules)
        }
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Tree depth: {metrics['tree_depth']}")
        logger.info(f"Number of leaves: {metrics['num_leaves']}")
        logger.info(f"Number of extracted rules: {metrics['num_rules']}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Print some rules
        logger.info("\nSample extracted rules:")
        for i, rule in enumerate(self.rules[:5]):
            logger.info(f"\nRule {i+1}: {rule['conditions']}")
            logger.info(f"  → Prediction: {rule['prediction']}")
            logger.info(f"  → Confidence: {rule['confidence']:.3f}")
        
        logger.info("=" * 80)
        logger.info("PREM TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return metrics
    
    def _extract_rules(self) -> List[Dict]:
        """
        Extract interpretable if-then rules from trained tree.
        
        Returns:
            List of rule dictionaries
        """
        if not self.is_trained:
            return []
        
        tree = self.model.tree_
        feature_names = self.feature_names
        class_names = ['Low', 'Medium', 'High']
        
        rules = []
        
        def recurse(node, conditions):
            """Recursively traverse tree to extract rules"""
            if tree.feature[node] != -2:  # Not a leaf
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left branch
                left_conditions = conditions + [f"{name} <= {threshold:.3f}"]
                recurse(tree.children_left[node], left_conditions)
                
                # Right branch
                right_conditions = conditions + [f"{name} > {threshold:.3f}"]
                recurse(tree.children_right[node], right_conditions)
            else:
                # Leaf node - create rule
                class_counts = tree.value[node][0]
                predicted_class = np.argmax(class_counts)
                confidence = class_counts[predicted_class] / class_counts.sum()
                
                rule = {
                    'conditions': ' AND '.join(conditions) if conditions else 'Always',
                    'prediction': class_names[predicted_class],
                    'prediction_id': int(predicted_class),
                    'confidence': float(confidence),
                    'samples': int(class_counts.sum())
                }
                rules.append(rule)
        
        recurse(0, [])
        return rules
    
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix (N x d)
            
        Returns:
            Probability matrix (N x C)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_applicable_rules(self, X: np.ndarray) -> List[Dict]:
        """
        Get rules that apply to given samples.
        
        Args:
            X: Feature matrix (N x d)
            
        Returns:
            List of applicable rules for each sample
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get decision path
        decision_paths = self.model.decision_path(X)
        
        applicable_rules = []
        for i in range(X.shape[0]):
            path = decision_paths[i].toarray()[0]
            leaf_id = self.model.apply(X[i:i+1])[0]
            
            # Find rule corresponding to this leaf
            matching_rules = [r for r in self.rules if 
                            self.model.tree_.feature[leaf_id] == -2]  # Is leaf
            
            applicable_rules.append(matching_rules)
        
        return applicable_rules
    
    def get_learner_profile(self, X: np.ndarray) -> dict:
        """
        Get comprehensive learner profile with rule explanation.
        
        Args:
            X: Feature vector for single learner (d,) or (1, d)
            
        Returns:
            Dictionary with predictions and rule explanations
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        pred = self.predict(X)[0]
        proba = self.predict_proba(X)[0]
        
        # Get decision path
        path = self.model.decision_path(X)
        node_indicator = path.toarray()[0]
        
        # Extract path description
        path_description = []
        for node_id in np.where(node_indicator)[0]:
            if self.model.tree_.feature[node_id] != -2:  # Not leaf
                feature_idx = self.model.tree_.feature[node_id]
                threshold = self.model.tree_.threshold[node_id]
                feature_name = self.feature_names[feature_idx]
                
                if X[0, feature_idx] <= threshold:
                    path_description.append(f"{feature_name} <= {threshold:.3f}")
                else:
                    path_description.append(f"{feature_name} > {threshold:.3f}")
        
        profile = {
            'predicted_class': int(pred),
            'predicted_class_name': ['Low', 'Medium', 'High'][pred],
            'class_probabilities': {
                'Low': float(proba[0]),
                'Medium': float(proba[1]),
                'High': float(proba[2])
            },
            'confidence': float(np.max(proba)),
            'decision_path': path_description,
            'rule_explanation': ' AND '.join(path_description) if path_description else 'Default'
        }
        
        return profile
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance from the tree.
        
        Returns:
            Dictionary mapping features to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
        
        return importance_dict
    
    def export_rules_text(self) -> str:
        """
        Export rules as readable text.
        
        Returns:
            String with all rules formatted
        """
        if not self.is_trained:
            return "Model not trained"
        
        text = "PEDAGOGICAL RULES:\n"
        text += "=" * 80 + "\n\n"
        
        for i, rule in enumerate(self.rules, 1):
            text += f"Rule {i}:\n"
            text += f"  IF {rule['conditions']}\n"
            text += f"  THEN Performance = {rule['prediction']}\n"
            text += f"  (Confidence: {rule['confidence']:.3f}, Samples: {rule['samples']})\n\n"
        
        return text
    
    def save_model(self, filepath: Optional[str] = None):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = MODELS_DIR / 'prem_model.pkl'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'classes': self.classes_,
            'feature_names': self.feature_names,
            'rules': self.rules
        }, filepath)
        
        logger.info(f"PREM model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None):
        """Load trained model from disk"""
        if filepath is None:
            filepath = MODELS_DIR / 'prem_model.pkl'
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        saved_data = joblib.load(filepath)
        
        self.model = saved_data['model']
        self.config = saved_data['config']
        self.classes_ = saved_data['classes']
        self.feature_names = saved_data['feature_names']
        self.rules = saved_data['rules']
        self.is_trained = True
        
        logger.info(f"PREM model loaded from {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PREM (PEDAGOGICAL RULE EXTRACTION MODEL) DEMO")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.50, 0.25])
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Feature names
    feature_names = [f"Feature_{i+1}" for i in range(n_features)]
    
    # Initialize and train PREM
    prem = PedagogicalRuleExtractionModel()
    metrics = prem.train(X_train, y_train, X_test, y_test, feature_names)
    
    # Print extracted rules
    print("\n" + "=" * 80)
    print("EXTRACTED RULES")
    print("=" * 80)
    print(prem.export_rules_text())
    
    # Get learner profile
    print("=" * 80)
    print("LEARNER PROFILE EXAMPLE")
    print("=" * 80)
    
    profile = prem.get_learner_profile(X_test[0])
    print("\nLearner Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance = prem.get_feature_importance()
    print("\nTop 5 Important Features:")
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.4f}")
    
    # Save model
    prem.save_model()
    
    print("\n" + "=" * 80)
    print("PREM Demo Complete!")
    print("=" * 80)
