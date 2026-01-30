"""
EduGen Preprocessing Module
Handles data cleaning, feature engineering, normalization, and SMOTE balancing
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats

from config import FEATURE_CONFIG, DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive preprocessing pipeline for student performance data.
    """
    
    def __init__(self, normalization='minmax'):
        """
        Initialize preprocessor.
        
        Args:
            normalization: Type of normalization ('minmax', 'zscore', 'robust')
        """
        self.normalization = normalization
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.quiz_columns = None
        
        # Initialize scaler based on type
        if normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalization == 'zscore':
            self.scaler = StandardScaler()
        elif normalization == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        logger.info(f"DataPreprocessor initialized with {normalization} normalization")
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'zero') -> pd.DataFrame:
        """
        Handle missing values and 'AB' (absent) markers in quiz scores.
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values ('zero', 'mean', 'median', 'forward_fill')
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Get quiz columns
        self.quiz_columns = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
        
        logger.info(f"Handling missing values using method: {method}")
        
        # Replace 'AB' with NaN first
        for col in self.quiz_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Apply chosen method
        if method == 'zero':
            df[self.quiz_columns] = df[self.quiz_columns].fillna(0)
        
        elif method == 'mean':
            # Class-wise mean imputation if Performance_Class exists
            if 'Performance_Class' in df.columns:
                for cls in df['Performance_Class'].unique():
                    mask = df['Performance_Class'] == cls
                    df.loc[mask, self.quiz_columns] = df.loc[mask, self.quiz_columns].fillna(
                        df.loc[mask, self.quiz_columns].mean()
                    )
            else:
                df[self.quiz_columns] = df[self.quiz_columns].fillna(df[self.quiz_columns].mean())
        
        elif method == 'median':
            if 'Performance_Class' in df.columns:
                for cls in df['Performance_Class'].unique():
                    mask = df['Performance_Class'] == cls
                    df.loc[mask, self.quiz_columns] = df.loc[mask, self.quiz_columns].fillna(
                        df.loc[mask, self.quiz_columns].median()
                    )
            else:
                df[self.quiz_columns] = df[self.quiz_columns].fillna(df[self.quiz_columns].median())
        
        elif method == 'forward_fill':
            df[self.quiz_columns] = df[self.quiz_columns].fillna(method='ffill').fillna(0)
        
        # Handle other numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        logger.info(f"Missing values handled. Remaining NaN count: {df.isnull().sum().sum()}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features as described in the paper.
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        
        logger.info("Engineering features...")
        
        # Ensure quiz columns are numeric
        if self.quiz_columns is None:
            self.quiz_columns = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
        
        quiz_scores = df[self.quiz_columns].values
        
        # 1. Top 9 Sum: Sum of best 9 quiz scores (as per paper)
        top_k = DATA_CONFIG.get('top_k_quizzes', 9)
        df['Top_9_Sum'] = np.apply_along_axis(
            lambda x: np.sort(x[~np.isnan(x)])[-top_k:].sum() if len(x[~np.isnan(x)]) >= top_k else x[~np.isnan(x)].sum(),
            axis=1,
            arr=quiz_scores
        )
        
        # 2. Average Score
        df['Avg_Score'] = df[self.quiz_columns].mean(axis=1)
        
        # 3. Consistency Index: Standard deviation across quizzes (stability measure)
        df['Consistency_Index'] = df[self.quiz_columns].std(axis=1)
        
        # 4. Final Score Normalized
        max_possible = DATA_CONFIG['max_quiz_score'] * top_k
        df['Final_Score_Normalized'] = (df['Top_9_Sum'] / max_possible) * 100
        
        # 5. Attendance Rate (if not already present)
        if 'Attendance' in df.columns and 'Attendance_Rate' not in df.columns:
            df['Attendance_Rate'] = df['Attendance']
        
        # 6. Score Trend: Linear regression slope of quiz scores over time
        def compute_trend(scores):
            """Compute linear trend of scores"""
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) < 2:
                return 0.0
            x = np.arange(len(valid_scores))
            slope, _, _, _, _ = stats.linregress(x, valid_scores)
            return slope
        
        df['Score_Trend'] = np.apply_along_axis(compute_trend, axis=1, arr=quiz_scores)
        
        # 7. Improvement Rate: (Last 3 quizzes avg - First 3 quizzes avg)
        if len(self.quiz_columns) >= 6:
            first_3 = df[self.quiz_columns[:3]].mean(axis=1)
            last_3 = df[self.quiz_columns[-3:]].mean(axis=1)
            df['Improvement_Rate'] = last_3 - first_3
        else:
            df['Improvement_Rate'] = 0.0
        
        # 8. Variability: Coefficient of variation
        mean_scores = df[self.quiz_columns].mean(axis=1)
        std_scores = df[self.quiz_columns].std(axis=1)
        df['Variability'] = np.where(mean_scores > 0, std_scores / mean_scores, 0)
        
        # 9. Pass Rate: Percentage of quizzes with score >= 3 (60%)
        passing_threshold = DATA_CONFIG['max_quiz_score'] * 0.6
        df['Pass_Rate'] = (df[self.quiz_columns] >= passing_threshold).sum(axis=1) / len(self.quiz_columns) * 100
        
        # 10. Max Score: Highest quiz score
        df['Max_Score'] = df[self.quiz_columns].max(axis=1)
        
        # 11. Min Score: Lowest quiz score
        df['Min_Score'] = df[self.quiz_columns].min(axis=1)
        
        # 12. Range: Difference between max and min
        df['Score_Range'] = df['Max_Score'] - df['Min_Score']
        
        logger.info(f"Engineered {12} new features")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, target_col: str = 'Performance_Class') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Encode categorical target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (DataFrame, encoded_labels)
        """
        df = df.copy()
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return df, None
        
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded = self.label_encoder.fit_transform(df[target_col])
        else:
            encoded = self.label_encoder.transform(df[target_col])
        
        logger.info(f"Encoded {target_col}: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return df, encoded
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None, 
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: Input DataFrame
            feature_cols: Columns to normalize. If None, normalizes all numeric columns.
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        if feature_cols is None:
            # Get all numeric columns except the target
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col not in ['Roll_No', 'Performance_Class']]
        
        logger.info(f"Normalizing {len(feature_cols)} features using {self.normalization}")
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        self.feature_names = feature_cols
        
        return df
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray, 
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) for class balancing.
        
        Args:
            X: Feature matrix
            y: Target labels
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if random_state is None:
            random_state = DATA_CONFIG['random_seed']
        
        logger.info(f"Applying SMOTE. Original class distribution: {np.bincount(y)}")
        
        # Determine k_neighbors based on smallest class size
        min_class_size = min(np.bincount(y))
        k_neighbors = min(5, min_class_size - 1)
        
        if k_neighbors < 1:
            logger.warning("Not enough samples for SMOTE. Skipping...")
            return X, y
        
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"After SMOTE class distribution: {np.bincount(y_resampled)}")
        logger.info(f"Dataset size: {len(y)} -> {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                           is_training: bool = True,
                           apply_smote_flag: bool = True,
                           target_col: str = 'Performance_Class') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (affects fitting)
            apply_smote_flag: Whether to apply SMOTE
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is target (None if not present)
        """
        logger.info("=" * 80)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df, method=FEATURE_CONFIG['handle_missing'])
        
        # Step 2: Engineer features
        df = self.engineer_features(df)
        
        # Step 3: Encode categorical target
        y = None
        if target_col in df.columns:
            df, y = self.encode_categorical(df, target_col)
        
        # Step 4: Select features for modeling
        # Exclude ID and target columns
        exclude_cols = ['Roll_No', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols and col not in self.quiz_columns]
        
        # Also include quiz columns for some models
        all_feature_cols = self.quiz_columns + feature_cols
        
        logger.info(f"Selected {len(all_feature_cols)} features for modeling")
        
        # Step 5: Normalize features
        df = self.normalize_features(df, all_feature_cols, fit=is_training)
        
        # Extract feature matrix
        X = df[all_feature_cols].values
        
        # Step 6: Apply SMOTE (only for training data)
        if is_training and apply_smote_flag and y is not None:
            X, y = self.apply_smote(X, y)
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Feature matrix shape: {X.shape}")
        if y is not None:
            logger.info(f"Target shape: {y.shape}")
            logger.info(f"Class distribution: {np.bincount(y)}")
        logger.info("=" * 80)
        
        return X, y
    
    def get_feature_importance_names(self) -> List[str]:
        """Get names of features used in modeling"""
        return self.feature_names if self.feature_names else []
    
    def inverse_transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized features back to original scale.
        
        Args:
            X: Normalized feature matrix
            
        Returns:
            Original scale features
        """
        return self.scaler.inverse_transform(X)


class FeatureSelector:
    """
    Feature selection utilities for identifying most important features.
    """
    
    @staticmethod
    def compute_feature_importance(X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str],
                                   method: str = 'mutual_info') -> Dict[str, float]:
        """
        Compute feature importance scores.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            method: Method for computing importance ('mutual_info', 'chi2')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.feature_selection import mutual_info_classif, chi2
        from sklearn.preprocessing import MinMaxScaler
        
        logger.info(f"Computing feature importance using {method}")
        
        if method == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=42)
        elif method == 'chi2':
            # Chi2 requires non-negative features
            X_positive = MinMaxScaler().fit_transform(X)
            scores, _ = chi2(X_positive, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        importance_dict = dict(zip(feature_names, scores))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True))
        
        logger.info("Top 10 most important features:")
        for i, (feat, score) in enumerate(list(sorted_importance.items())[:10], 1):
            logger.info(f"  {i}. {feat}: {score:.4f}")
        
        return sorted_importance


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    from data_loader import StudentDataLoader
    
    print("=" * 80)
    print("EDUGEN PREPROCESSING DEMO")
    print("=" * 80)
    
    # Load data
    loader = StudentDataLoader()
    data = loader.load_data()
    train_df, val_df, test_df = loader.split_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(normalization='minmax')
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    X_train, y_train = preprocessor.preprocess_pipeline(
        train_df, 
        is_training=True, 
        apply_smote_flag=True
    )
    
    # Preprocess validation data
    print("\nPreprocessing validation data...")
    X_val, y_val = preprocessor.preprocess_pipeline(
        val_df, 
        is_training=False, 
        apply_smote_flag=False
    )
    
    # Preprocess test data
    print("\nPreprocessing test data...")
    X_test, y_test = preprocessor.preprocess_pipeline(
        test_df, 
        is_training=False, 
        apply_smote_flag=False
    )
    
    print("\n" + "=" * 80)
    print("PREPROCESSING RESULTS")
    print("=" * 80)
    print(f"Training set: X shape {X_train.shape}, y shape {y_train.shape}")
    print(f"Validation set: X shape {X_val.shape}, y shape {y_val.shape}")
    print(f"Test set: X shape {X_test.shape}, y shape {y_test.shape}")
    
    # Compute feature importance
    feature_names = preprocessor.get_feature_importance_names()
    if feature_names:
        importance = FeatureSelector.compute_feature_importance(
            X_train, y_train, feature_names
        )
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
