import numpy as np
import logging
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

from config import LBSM_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearnerBehaviourSegmentationModule:
    """
    LBSM using K-Means clustering to discover behavioral archetypes.
    Minimizes: J = Σ_j Σ_{x_i ∈ C_j} ||x_i - μ_j||^2
    """
    
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = LBSM_CONFIG
        
        self.config = config
        self.model = None
        self.is_trained = False
        self.cluster_centers = None
        self.cluster_labels = None
        
        self._initialize_model()
        
        logger.info(f"LBSM initialized with {config['n_clusters']} clusters")
    
    def _initialize_model(self):
        self.model = KMeans(
            n_clusters=self.config['n_clusters'],
            init=self.config['init'],
            max_iter=self.config['max_iter'],
            n_init=self.config['n_init'],
            random_state=self.config['random_state'],
            algorithm=self.config['algorithm']
        )
    
    def train(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None):
        logger.info("=" * 80)
        logger.info("TRAINING LEARNER BEHAVIOUR SEGMENTATION MODULE (LBSM)")
        logger.info("=" * 80)
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Feature dimension: {X_train.shape[1]}")
        
        # Fit K-Means
        self.model.fit(X_train)
        
        self.is_trained = True
        self.cluster_centers = self.model.cluster_centers_
        self.cluster_labels = self.model.labels_
        
        # Compute metrics
        inertia = self.model.inertia_
        silhouette = silhouette_score(X_train, self.cluster_labels)
        
        metrics = {
            'inertia': inertia,
            'silhouette_score': silhouette,
            'n_clusters': self.config['n_clusters'],
            'cluster_sizes': np.bincount(self.cluster_labels)
        }
        
        logger.info(f"Inertia: {inertia:.2f}")
        logger.info(f"Silhouette score: {silhouette:.4f}")
        logger.info(f"Cluster sizes: {metrics['cluster_sizes']}")
        
        logger.info("=" * 80)
        logger.info("LBSM TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def get_learner_profile(self, X: np.ndarray) -> dict:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        cluster_id = self.predict(X)[0]
        distance_to_center = np.linalg.norm(X - self.cluster_centers[cluster_id])
        
        profile = {
            'cluster_id': int(cluster_id),
            'cluster_name': ['Low Performers', 'Medium Performers', 'High Performers'][cluster_id],
            'distance_to_center': float(distance_to_center),
            'is_typical': distance_to_center < np.median(np.linalg.norm(
                self.cluster_centers - self.cluster_centers[cluster_id], axis=1))
        }
        
        return profile
    
    def save_model(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = MODELS_DIR / 'lbsm_model.pkl'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'cluster_centers': self.cluster_centers
        }, filepath)
        
        logger.info(f"LBSM model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = MODELS_DIR / 'lbsm_model.pkl'
        
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.config = saved_data['config']
        self.cluster_centers = saved_data['cluster_centers']
        self.is_trained = True
        
        logger.info(f"LBSM model loaded from {filepath}")


if __name__ == '__main__':
    # Demo code
    np.random.seed(42)
    X = np.random.randn(500, 10)
    
    lbsm = LearnerBehaviourSegmentationModule()
    metrics = lbsm.train(X)
    
    profile = lbsm.get_learner_profile(X[0])
    print("Learner Profile:", profile)
    
    lbsm.save_model()


