"""
EduGen Data Loader Module
Handles loading, generation, and management of student performance datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
from sklearn.model_selection import train_test_split

from config import (
    DATA_CONFIG, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentDataLoader:
    """
    Comprehensive data loader for student performance records.
    Handles both real and synthetic dataset generation.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the dataset CSV file. If None, will look for default location.
        """
        self.data_path = data_path
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        logger.info("StudentDataLoader initialized")
    
    def generate_synthetic_data(self, 
                                num_students: Optional[int] = None,
                                save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic student performance data for testing and development.
        
        Args:
            num_students: Number of student records to generate
            save_path: Path to save generated data
            
        Returns:
            DataFrame with synthetic student data
        """
        if num_students is None:
            num_students = SYNTHETIC_DATA_CONFIG['num_students']
        
        num_quizzes = SYNTHETIC_DATA_CONFIG['num_quizzes']
        
        logger.info(f"Generating synthetic data for {num_students} students")
        
        # Generate student IDs
        student_ids = [f"S{str(i+1).zfill(4)}" for i in range(num_students)]
        
        # Generate performance classes based on distribution
        perf_dist = SYNTHETIC_DATA_CONFIG['performance_distribution']
        classes = np.random.choice(
            list(perf_dist.keys()),
            size=num_students,
            p=list(perf_dist.values())
        )
        
        # Initialize data dictionary
        data_dict = {'Roll_No': student_ids}
        
        # Generate quiz scores based on performance class
        for i in range(1, num_quizzes + 1):
            quiz_scores = []
            for cls in classes:
                if cls == 'High':
                    # High performers: mean 4.0, std 0.8
                    score = np.random.normal(4.0, 0.8)
                elif cls == 'Medium':
                    # Medium performers: mean 3.0, std 1.0
                    score = np.random.normal(3.0, 1.0)
                else:  # Low
                    # Low performers: mean 1.5, std 1.0
                    score = np.random.normal(1.5, 1.0)
                
                # Clip scores to valid range [0, 5]
                score = np.clip(score, 0, 5)
                
                # Randomly mark some as absent (5% chance)
                if np.random.random() < 0.05:
                    score = 'AB'
                else:
                    score = round(score, 1)
                
                quiz_scores.append(score)
            
            data_dict[f'Q{i}'] = quiz_scores
        
        # Generate attendance (correlation with performance)
        attendance = []
        for cls in classes:
            if cls == 'High':
                att = np.random.normal(85, 10)
            elif cls == 'Medium':
                att = np.random.normal(70, 15)
            else:
                att = np.random.normal(55, 20)
            attendance.append(np.clip(att, 0, 100))
        
        data_dict['Attendance'] = attendance
        
        # Generate study hours
        study_hours = []
        for cls in classes:
            if cls == 'High':
                hours = np.random.normal(18, 4)
            elif cls == 'Medium':
                hours = np.random.normal(12, 5)
            else:
                hours = np.random.normal(8, 4)
            study_hours.append(max(0, hours))
        
        data_dict['Study_Hours'] = study_hours
        
        # Generate behavior scores (1-5 scale)
        behavior_scores = []
        for cls in classes:
            if cls == 'High':
                score = np.random.normal(4.5, 0.5)
            elif cls == 'Medium':
                score = np.random.normal(3.5, 0.8)
            else:
                score = np.random.normal(2.5, 1.0)
            behavior_scores.append(np.clip(score, 1, 5))
        
        data_dict['Behaviour_Score'] = behavior_scores
        
        # Add performance class
        data_dict['Performance_Class'] = classes
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Save if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Synthetic data saved to {save_path}")
        
        logger.info(f"Generated {len(df)} student records")
        logger.info(f"Class distribution: {df['Performance_Class'].value_counts().to_dict()}")
        
        return df
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load student performance data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, uses instance data_path.
            
        Returns:
            DataFrame with student data
        """
        if file_path is None:
            file_path = self.data_path
        
        if file_path is None:
            # Try default locations
            default_paths = [
                RAW_DATA_DIR / 'student_performance.csv',
                RAW_DATA_DIR / 'edugen_dataset.csv',
            ]
            
            for path in default_paths:
                if path.exists():
                    file_path = path
                    break
            
            if file_path is None:
                logger.warning("No data file found. Generating synthetic data...")
                return self.generate_synthetic_data(
                    save_path=RAW_DATA_DIR / 'synthetic_student_data.csv'
                )
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
            logger.info(f"Columns: {list(self.data.columns)}")
            return self.data
        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            logger.info("Generating synthetic data instead...")
            return self.generate_synthetic_data(
                save_path=RAW_DATA_DIR / 'synthetic_student_data.csv'
            )
    
    def split_data(self, 
                   data: Optional[pd.DataFrame] = None,
                   stratify_col: str = 'Performance_Class') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: DataFrame to split. If None, uses loaded data.
            stratify_col: Column to use for stratified splitting
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            data = self.data
        
        train_split = DATA_CONFIG['train_split']
        val_split = DATA_CONFIG['val_split']
        test_split = DATA_CONFIG['test_split']
        random_seed = DATA_CONFIG['random_seed']
        
        # Verify splits sum to 1.0
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Train, val, and test splits must sum to 1.0"
        
        logger.info(f"Splitting data: {train_split:.0%} train, {val_split:.0%} val, {test_split:.0%} test")
        
        # Stratify by performance class if column exists
        stratify = data[stratify_col] if stratify_col in data.columns else None
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_split,
            random_state=random_seed,
            stratify=stratify
        )
        
        # Second split: separate train and validation
        val_ratio = val_split / (train_split + val_split)
        stratify_train_val = train_val_data[stratify_col] if stratify_col in train_val_data.columns else None
        
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=stratify_train_val
        )
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        logger.info(f"Train set: {len(train_data)} samples")
        logger.info(f"Validation set: {len(val_data)} samples")
        logger.info(f"Test set: {len(test_data)} samples")
        
        # Log class distributions
        if stratify_col in data.columns:
            logger.info("\nClass distributions:")
            logger.info(f"Train: {train_data[stratify_col].value_counts().to_dict()}")
            logger.info(f"Val: {val_data[stratify_col].value_counts().to_dict()}")
            logger.info(f"Test: {test_data[stratify_col].value_counts().to_dict()}")
        
        return train_data, val_data, test_data
    
    def get_statistics(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compute basic statistics for the dataset.
        
        Args:
            data: DataFrame to analyze. If None, uses loaded data.
            
        Returns:
            Dictionary with dataset statistics
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            data = self.data
        
        stats = {
            'num_records': len(data),
            'num_features': len(data.columns),
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'ab_counts': {},
        }
        
        # Count 'AB' (absent) values in quiz columns
        quiz_cols = [col for col in data.columns if col.startswith('Q')]
        for col in quiz_cols:
            ab_count = (data[col] == 'AB').sum() if col in data.columns else 0
            stats['ab_counts'][col] = ab_count
        
        # Class distribution
        if 'Performance_Class' in data.columns:
            stats['class_distribution'] = data['Performance_Class'].value_counts().to_dict()
        
        # Numerical statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        stats['numeric_stats'] = data[numeric_cols].describe().to_dict()
        
        return stats
    
    def save_splits(self, output_dir: Optional[Path] = None):
        """
        Save train, validation, and test splits to separate CSV files.
        
        Args:
            output_dir: Directory to save split files
        """
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.train_data is not None:
            train_path = output_dir / 'train_data.csv'
            self.train_data.to_csv(train_path, index=False)
            logger.info(f"Saved train data to {train_path}")
        
        if self.val_data is not None:
            val_path = output_dir / 'val_data.csv'
            self.val_data.to_csv(val_path, index=False)
            logger.info(f"Saved validation data to {val_path}")
        
        if self.test_data is not None:
            test_path = output_dir / 'test_data.csv'
            self.test_data.to_csv(test_path, index=False)
            logger.info(f"Saved test data to {test_path}")
    
    def load_splits(self, data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved train, validation, and test splits.
        
        Args:
            data_dir: Directory containing split files
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if data_dir is None:
            data_dir = PROCESSED_DATA_DIR
        
        data_dir = Path(data_dir)
        
        train_path = data_dir / 'train_data.csv'
        val_path = data_dir / 'val_data.csv'
        test_path = data_dir / 'test_data.csv'
        
        self.train_data = pd.read_csv(train_path)
        self.val_data = pd.read_csv(val_path)
        self.test_data = pd.read_csv(test_path)
        
        logger.info(f"Loaded train data: {len(self.train_data)} samples")
        logger.info(f"Loaded validation data: {len(self.val_data)} samples")
        logger.info(f"Loaded test data: {len(self.test_data)} samples")
        
        return self.train_data, self.val_data, self.test_data


def create_knowledge_base_samples():
    """
    Create sample knowledge base entries for RAG.
    These would typically come from syllabi, textbooks, etc.
    """
    knowledge_base = {
        'Mathematics': [
            {
                'topic': 'Calculus',
                'content': 'Calculus is the mathematical study of continuous change. '
                          'It has two major branches: differential calculus (concerning rates of change and slopes of curves) '
                          'and integral calculus (concerning accumulation of quantities and areas under curves).',
                'difficulty': 'Medium',
                'bloom_level': 'Remember'
            },
            {
                'topic': 'Linear Algebra',
                'content': 'Linear algebra is the branch of mathematics concerning linear equations, '
                          'linear functions, and their representations through matrices and vector spaces. '
                          'Matrix multiplication is performed by taking the dot product of rows and columns.',
                'difficulty': 'Medium',
                'bloom_level': 'Apply'
            },
            {
                'topic': 'Probability',
                'content': 'Probability theory is the branch of mathematics that studies random phenomena. '
                          'The probability of an event is a number between 0 and 1, where 0 indicates impossibility '
                          'and 1 indicates certainty. The sum of probabilities of all possible outcomes equals 1.',
                'difficulty': 'Easy',
                'bloom_level': 'Remember'
            },
        ],
        'Physics': [
            {
                'topic': 'Newton\'s Laws',
                'content': 'Newton\'s First Law states that an object at rest stays at rest and an object in motion '
                          'stays in motion with the same speed and direction unless acted upon by an external force. '
                          'This is also known as the law of inertia.',
                'difficulty': 'Easy',
                'bloom_level': 'Remember'
            },
            {
                'topic': 'Thermodynamics',
                'content': 'The First Law of Thermodynamics states that energy cannot be created or destroyed, '
                          'only transformed from one form to another. This is expressed as ΔU = Q - W, '
                          'where ΔU is change in internal energy, Q is heat added, and W is work done.',
                'difficulty': 'Hard',
                'bloom_level': 'Analyze'
            },
        ],
    }
    
    # Save knowledge base
    kb_path = PROCESSED_DATA_DIR / 'knowledge_base.csv'
    
    # Flatten to DataFrame
    records = []
    for subject, topics in knowledge_base.items():
        for item in topics:
            records.append({
                'subject': subject,
                'topic': item['topic'],
                'content': item['content'],
                'difficulty': item['difficulty'],
                'bloom_level': item['bloom_level']
            })
    
    kb_df = pd.DataFrame(records)
    kb_df.to_csv(kb_path, index=False)
    logger.info(f"Saved knowledge base to {kb_path}")
    
    return kb_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Demo usage
    print("=" * 80)
    print("EDUGEN DATA LOADER DEMO")
    print("=" * 80)
    
    # Initialize loader
    loader = StudentDataLoader()
    
    # Generate or load data
    data = loader.load_data()
    
    # Display basic info
    print("\nDataset Info:")
    print(data.info())
    
    print("\nFirst few records:")
    print(data.head())
    
    # Get statistics
    stats = loader.get_statistics()
    print("\nDataset Statistics:")
    print(f"Total records: {stats['num_records']}")
    print(f"Total features: {stats['num_features']}")
    if 'class_distribution' in stats:
        print(f"Class distribution: {stats['class_distribution']}")
    
    # Split data
    train_df, val_df, test_df = loader.split_data()
    
    # Save splits
    loader.save_splits()
    
    # Create knowledge base
    print("\nCreating knowledge base...")
    kb_df = create_knowledge_base_samples()
    print(f"Knowledge base created with {len(kb_df)} entries")
    
    print("\n" + "=" * 80)
    print("Data loading complete!")
    print("=" * 80)
