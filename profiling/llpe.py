import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Optional
from pathlib import Path

from config import LLPE_CONFIG, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification"""
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class LatentLearningPatternExtractor:
    """
    LLPE using MLP for capturing non-linear learning patterns.
    P(y=j|x) = exp(z_j) / Σ_k exp(z_k)
    """
    
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = LLPE_CONFIG
        
        self.config = config
        self.model = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"LLPE initialized on device: {self.device}")
    
    def _initialize_model(self, input_dim, num_classes):
        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=self.config['hidden_layers'],
            num_classes=num_classes,
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None):
        
        logger.info("=" * 80)
        logger.info("TRAINING LATENT LEARNING PATTERN EXTRACTOR (LLPE)")
        logger.info("=" * 80)
        
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        self._initialize_model(input_dim, num_classes)
        
        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_X = torch.FloatTensor(X_val).to(self.device)
                    val_y = torch.LongTensor(y_val).to(self.device)
                    val_outputs = self.model(val_X)
                    val_loss = criterion(val_outputs, val_y).item()
                    val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                              f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                              f"Train Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses if val_losses else None,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None
        }
        
        logger.info("=" * 80)
        logger.info("LLPE TRAINING COMPLETE")
        logger.info("=" * 80)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """Extract latent embedding before final layer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before extraction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            # Get output from second-to-last layer
            for i, layer in enumerate(self.model.network):
                X_tensor = layer(X_tensor)
                if i == len(self.model.network) - 2:
                    break
        
        return X_tensor.cpu().numpy()
    
    def get_learner_profile(self, X: np.ndarray) -> dict:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        pred = self.predict(X)[0]
        proba = self.predict_proba(X)[0]
        embedding = self.get_embedding(X)[0]
        
        profile = {
            'predicted_class': int(pred),
            'predicted_class_name': ['Low', 'Medium', 'High'][pred],
            'class_probabilities': {
                'Low': float(proba[0]),
                'Medium': float(proba[1]),
                'High': float(proba[2])
            },
            'confidence': float(np.max(proba)),
            'latent_embedding': embedding.tolist(),
            'embedding_norm': float(np.linalg.norm(embedding))
        }
        
        return profile
    
    def save_model(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = MODELS_DIR / 'llpe_model.pth'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, filepath)
        
        logger.info(f"LLPE model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None, input_dim: int = None, num_classes: int = 3):
        if filepath is None:
            filepath = MODELS_DIR / 'llpe_model.pth'
        
        checkpoint = torch.load(filepath)
        self.config = checkpoint['config']
        
        if input_dim:
            self._initialize_model(input_dim, num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = True
        
        logger.info(f"LLPE model loaded from {filepath}")


if __name__ == '__main__':
    # Demo code
    np.random.seed(42)
    X_train = np.random.randn(400, 15)
    y_train = np.random.choice([0, 1, 2], 400)
    X_val = np.random.randn(100, 15)
    y_val = np.random.choice([0, 1, 2], 100)
    
    llpe = LatentLearningPatternExtractor()
    metrics = llpe.train(X_train, y_train, X_val, y_val)
    
    profile = llpe.get_learner_profile(X_val[0])
    print("Learner Profile:", profile)
    
  