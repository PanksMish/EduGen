"""
EduGen Configuration File
Contains all hyperparameters, paths, and system configurations
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
KNOWLEDGE_BASE_DIR = DATA_DIR / 'knowledge_base'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 KNOWLEDGE_BASE_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_CONFIG = {
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
    'num_quizzes': 12,
    'max_quiz_score': 5,
    'top_k_quizzes': 9,  # Top 9 sum feature
    'missing_value': 'AB',  # Absent marker
    'class_labels': ['Low', 'Medium', 'High'],
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
FEATURE_CONFIG = {
    'engineered_features': [
        'top_9_sum',
        'final_score_normalized',
        'consistency_index',
        'attendance_rate',
        'avg_score',
        'score_trend',
        'improvement_rate'
    ],
    'normalization': 'minmax',  # 'minmax', 'zscore', or 'robust'
    'handle_missing': 'zero',  # 'zero', 'mean', 'median', 'forward_fill'
}

# ============================================================================
# PROFILING MODELS CONFIGURATION
# ============================================================================

# Performance Probability Estimator (PPE) - Logistic Regression
PPE_CONFIG = {
    'model_type': 'logistic_regression',
    'learning_rate': 0.01,
    'max_iter': 1000,
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'batch_size': 32,
    'early_stopping': True,
    'patience': 10,
}

# Pedagogical Rule Extraction Model (PREM) - Decision Tree
PREM_CONFIG = {
    'model_type': 'decision_tree',
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'criterion': 'gini',  # 'gini' or 'entropy'
    'max_features': None,
    'class_weight': 'balanced',
}

# Learner Behaviour Segmentation Module (LBSM) - K-Means
LBSM_CONFIG = {
    'model_type': 'kmeans',
    'n_clusters': 3,  # Low, Medium, High performers
    'init': 'k-means++',
    'max_iter': 300,
    'n_init': 10,
    'random_state': 42,
    'algorithm': 'lloyd',
}

# Latent Learning Pattern Extractor (LLPE) - MLP
LLPE_CONFIG = {
    'model_type': 'mlp',
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    'early_stopping': True,
    'patience': 10,
}

# ============================================================================
# PSYCHOMETRIC MODELING (IRT)
# ============================================================================
IRT_CONFIG = {
    'model': '2PL',  # Two-Parameter Logistic Model
    'max_iter': 100,
    'tol': 1e-4,
    'ability_range': (-3, 3),
    'difficulty_range': (-3, 3),
    'discrimination_range': (0.5, 2.5),
    'initial_ability': 0.0,
    'ability_thresholds': {
        'easy': -1.0,      # θ < -1.0
        'medium': 1.0,     # -1.0 ≤ θ < 1.0
        'hard': float('inf')  # θ ≥ 1.0
    }
}

# ============================================================================
# BLOOM'S TAXONOMY CONFIGURATION
# ============================================================================
BLOOM_CONFIG = {
    'levels': {
        1: 'Remember',
        2: 'Apply',
        3: 'Analyze',
        4: 'Evaluate',  # Extended levels
        5: 'Create',
    },
    'active_levels': [1, 2, 3],  # Paper uses 3 levels
    'default_level': 1,
    'level_weights': {
        1: 0.3,  # Remember
        2: 0.4,  # Apply
        3: 0.3,  # Analyze
    },
    'keywords': {
        1: ['define', 'list', 'recall', 'recognize', 'identify', 'name'],
        2: ['apply', 'demonstrate', 'calculate', 'solve', 'use', 'implement'],
        3: ['analyze', 'compare', 'contrast', 'examine', 'differentiate', 'investigate'],
    }
}

# ============================================================================
# RAG (RETRIEVAL-AUGMENTED GENERATION) CONFIGURATION
# ============================================================================
RAG_CONFIG = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'vector_db': 'faiss',  # 'faiss' or 'chroma'
    'top_k_retrieval': 5,
    'chunk_size': 512,
    'chunk_overlap': 50,
    'similarity_metric': 'cosine',
    'rerank': True,
    'rerank_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
}

# ============================================================================
# QUESTION GENERATION CONFIGURATION
# ============================================================================

# LLM-based Question Generation (Large Model)
LLM_QG_CONFIG = {
    'model_name': 'gpt-3.5-turbo',  # or 'gpt-4', 'claude-2', etc.
    'temperature': 0.7,
    'max_tokens': 512,
    'top_p': 0.9,
    'frequency_penalty': 0.5,
    'presence_penalty': 0.3,
    'num_candidates': 5,  # Generate multiple candidates
    'use_rag': True,
    'enforce_bloom': True,
    'enforce_difficulty': True,
}

# SLM-based Question Generation (Small Model - Efficient)
SLM_QG_CONFIG = {
    'model_name': 'gpt2-medium',  # or other lightweight models
    'temperature': 0.8,
    'max_tokens': 256,
    'top_p': 0.95,
    'num_candidates': 3,
    'use_rag': True,
    'enforce_bloom': True,
    'enforce_difficulty': True,
}

# Question Filtering and Validation
QUESTION_FILTER_CONFIG = {
    'min_length': 20,
    'max_length': 300,
    'require_question_mark': True,
    'check_factual_consistency': True,
    'consistency_threshold': 0.7,
    'check_bloom_alignment': True,
    'bloom_confidence_threshold': 0.6,
    'check_difficulty_match': True,
    'difficulty_tolerance': 0.5,
}

# ============================================================================
# ALGORITHM CONFIGURATIONS
# ============================================================================

# APDEA: Adaptive Profiling and Difficulty Estimation Algorithm
APDEA_CONFIG = {
    'fusion_weights': {
        'probability': 0.3,    # λ1 for PPE
        'rules': 0.2,          # λ2 for PREM
        'clusters': 0.2,       # λ3 for LBSM
        'latent': 0.3,         # λ4 for LLPE
    },
    'bloom_prediction_method': 'weighted_voting',
    'update_frequency': 'per_question',  # or 'per_quiz'
}

# CTQS: Context-Aware Transformer-RAG Question Synthesis
CTQS_CONFIG = {
    'max_retries': 3,
    'psychometric_filtering': True,
    'compatibility_threshold': 1.0,  # |θ - b| threshold
    'factual_checking': True,
    'generation_timeout': 30,  # seconds
}

# BSSCA: Benchmark and Scholarship Suitability Computation Algorithm
BSSCA_CONFIG = {
    'benchmark_weights': {
        'performance': 0.35,   # ω1
        'stability': 0.25,     # ω2
        'cluster': 0.20,       # ω3
        'latent': 0.20,        # ω4
    },
    'scholarship_weights': {
        'benchmark': 0.4,      # η1
        'ability': 0.35,       # η2
        'latent_strength': 0.25,  # η3
    },
    'cluster_advantages': {
        0: 0.3,  # Low performer cluster
        1: 0.6,  # Medium performer cluster
        2: 1.0,  # High performer cluster
    },
    'scholarship_threshold': 0.75,  # Minimum score for scholarship consideration
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVALUATION_CONFIG = {
    'metrics': [
        'pedagogical_success_rate',
        'precision',
        'recall',
        'f1_score',
        'accuracy',
        'hallucination_rate',
        'cognitive_coverage',
        'calibration_error',
        'efficiency',
    ],
    'num_bins_ece': 10,  # For Expected Calibration Error
    'confidence_level': 0.95,
}

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_CONFIG = {
    'num_runs': 5,  # Multiple runs for averaging
    'cross_validation_folds': 5,
    'save_predictions': True,
    'save_models': True,
    'generate_plots': True,
    'verbose': True,
}

# ============================================================================
# COMPUTATIONAL RESOURCES
# ============================================================================
COMPUTE_CONFIG = {
    'device': 'cuda',  # 'cuda', 'cpu', or 'mps' (for Mac)
    'num_workers': 4,
    'pin_memory': True,
    'deterministic': True,
}

# ============================================================================
# LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': LOGS_DIR / 'edugen.log',
}

# ============================================================================
# API KEYS (Load from environment)
# ============================================================================
API_KEYS = {
    'openai': os.getenv('OPENAI_API_KEY', ''),
    'anthropic': os.getenv('ANTHROPIC_API_KEY', ''),
    'huggingface': os.getenv('HUGGINGFACE_API_KEY', ''),
}

# ============================================================================
# SYNTHETIC DATA GENERATION (for testing)
# ============================================================================
SYNTHETIC_DATA_CONFIG = {
    'num_students': 1053,
    'num_quizzes': 12,
    'performance_distribution': {
        'Low': 0.25,
        'Medium': 0.50,
        'High': 0.25,
    },
    'attendance_mean': 75.0,
    'attendance_std': 15.0,
    'study_hours_mean': 12.0,
    'study_hours_std': 5.0,
    'noise_level': 0.1,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(config_name):
    """Retrieve a specific configuration dictionary"""
    config_map = {
        'data': DATA_CONFIG,
        'feature': FEATURE_CONFIG,
        'ppe': PPE_CONFIG,
        'prem': PREM_CONFIG,
        'lbsm': LBSM_CONFIG,
        'llpe': LLPE_CONFIG,
        'irt': IRT_CONFIG,
        'bloom': BLOOM_CONFIG,
        'rag': RAG_CONFIG,
        'llm_qg': LLM_QG_CONFIG,
        'slm_qg': SLM_QG_CONFIG,
        'question_filter': QUESTION_FILTER_CONFIG,
        'apdea': APDEA_CONFIG,
        'ctqs': CTQS_CONFIG,
        'bssca': BSSCA_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'compute': COMPUTE_CONFIG,
        'logging': LOGGING_CONFIG,
        'synthetic': SYNTHETIC_DATA_CONFIG,
    }
    return config_map.get(config_name, {})

def print_config():
    """Print all configurations"""
    print("=" * 80)
    print("EDUGEN FRAMEWORK CONFIGURATION")
    print("=" * 80)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("\nData Split:", DATA_CONFIG['train_split'], 
          DATA_CONFIG['val_split'], DATA_CONFIG['test_split'])
    print("Random Seed:", DATA_CONFIG['random_seed'])
    print("\nActive Bloom Levels:", BLOOM_CONFIG['active_levels'])
    print("IRT Model:", IRT_CONFIG['model'])
    print("=" * 80)

if __name__ == '__main__':
    print_config()
