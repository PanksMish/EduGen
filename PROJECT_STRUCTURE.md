# EduGen Project Structure

Complete implementation of the paper: "A Pedagogy-Guided Transformer-RAG Framework for Adaptive Question Generation and Learner Profiling"

## 📁 Complete File Structure

```
edugen_implementation/
│
├── README.md                          ✅ Main project documentation
├── requirements.txt                   ✅ Python dependencies
├── config.py                          ✅ Configuration settings
├── setup.py                           ⚠️  Package installation script
├── LICENSE                            ⚠️  MIT License
│
├── data/                              📊 Data directory
│   ├── raw/                          # Raw dataset files
│   │   └── student_performance.csv
│   ├── processed/                    # Preprocessed data
│   │   ├── train_data.csv
│   │   ├── val_data.csv
│   │   ├── test_data.csv
│   │   └── knowledge_base.csv
│   └── knowledge_base/               # RAG knowledge base
│
├── data_loader.py                     ✅ Dataset loading utilities
├── preprocessing.py                   ✅ Data preprocessing pipeline
│
├── profiling/                         🧠 Learner Profiling Models
│   ├── __init__.py                   ✅
│   ├── ppe.py                        ✅ Performance Probability Estimator
│   ├── prem.py                       ⚠️  Pedagogical Rule Extraction Model
│   ├── lbsm.py                       ⚠️  Behaviour Segmentation Module
│   └── llpe.py                       ⚠️  Latent Learning Pattern Extractor
│
├── psychometric/                      📐 Psychometric Modeling
│   ├── __init__.py                   ⚠️
│   └── irt_model.py                  ⚠️  Item Response Theory implementation
│
├── generation/                        🤖 Question Generation
│   ├── __init__.py                   ⚠️
│   ├── rag_engine.py                 ⚠️  Retrieval-Augmented Generation
│   ├── question_generator.py         ⚠️  Question synthesis
│   └── bloom_classifier.py           ⚠️  Bloom's Taxonomy alignment
│
├── algorithms/                        🔬 Core Algorithms
│   ├── __init__.py                   ⚠️
│   ├── apdea.py                      ⚠️  Algorithm: Adaptive Profiling & Difficulty Estimation
│   ├── ctqs.py                       ⚠️  Algorithm: Context-Aware Question Synthesis
│   └── bssca.py                      ⚠️  Algorithm: Benchmark & Scholarship Computation
│
├── evaluation/                        📊 Evaluation Metrics
│   ├── __init__.py                   ⚠️
│   └── metrics.py                    ⚠️  PSR, F1, Calibration, etc.
│
├── utils/                             🛠️ Utility Functions
│   ├── __init__.py                   ⚠️
│   └── helpers.py                    ⚠️  Helper functions
│
├── experiments/                       🧪 Experimental Scripts
│   ├── run_experiments.py            ⚠️  Main experiment runner
│   ├── ablation_study.py             ⚠️  Ablation experiments
│   ├── pilot_study.py                ⚠️  Empirical pilot analysis
│   └── visualize_results.py          ⚠️  Result visualization
│
├── models/                            💾 Saved Models
│   ├── ppe_model.pkl
│   ├── prem_model.pkl
│   ├── lbsm_model.pkl
│   └── llpe_model.pth
│
├── results/                           📈 Experimental Results
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── notebooks/                         📓 Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_profiling_demo.ipynb
│   ├── 03_question_generation.ipynb
│   └── 04_full_pipeline.ipynb
│
├── tests/                             ✅ Unit Tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_profiling.py
│   ├── test_algorithms.py
│   └── test_generation.py
│
├── scripts/                           🔧 Utility Scripts
│   ├── download_models.py
│   ├── generate_synthetic_data.py
│   └── setup_knowledge_base.py
│
└── logs/                              📝 Log Files
    └── edugen.log

```

## ✅ Files Already Created

1. **README.md** - Complete project documentation
2. **requirements.txt** - All Python dependencies
3. **config.py** - Comprehensive configuration
4. **data_loader.py** - Data loading with synthetic generation
5. **preprocessing.py** - Full preprocessing pipeline
6. **profiling/ppe.py** - Performance Probability Estimator
7. **profiling/__init__.py** - Profiling module init

## ⚠️ Files to Be Created

### High Priority (Core Implementation)
- **profiling/prem.py** - Decision Tree based rule extraction
- **profiling/lbsm.py** - K-Means clustering for behavior
- **profiling/llpe.py** - MLP for latent patterns
- **psychometric/irt_model.py** - 2PL IRT implementation
- **algorithms/apdea.py** - Algorithm 5 from paper
- **algorithms/ctqs.py** - Algorithm 6 from paper
- **algorithms/bssca.py** - Algorithm 7 from paper

### Medium Priority (Generation & RAG)
- **generation/rag_engine.py** - RAG pipeline
- **generation/question_generator.py** - LLM/SLM question generation
- **generation/bloom_classifier.py** - Cognitive level classification

### Standard Priority (Evaluation & Utils)
- **evaluation/metrics.py** - PSR, F1, calibration metrics
- **utils/helpers.py** - Utility functions
- **experiments/run_experiments.py** - Main experiment script

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate/Load Data
```python
from data_loader import StudentDataLoader

loader = StudentDataLoader()
data = loader.load_data()  # Auto-generates if not found
train, val, test = loader.split_data()
loader.save_splits()
```

### 3. Preprocess Data
```python
from preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
X_train, y_train = preprocessor.preprocess_pipeline(train, is_training=True)
X_val, y_val = preprocessor.preprocess_pipeline(val, is_training=False)
```

### 4. Train Profiling Models
```python
from profiling import PerformanceProbabilityEstimator

ppe = PerformanceProbabilityEstimator()
ppe.train(X_train, y_train, X_val, y_val)
ppe.save_model()
```

### 5. Run Complete Pipeline
```python
# To be implemented in experiments/run_experiments.py
from experiments import run_full_pipeline

results = run_full_pipeline()
```

## 📊 Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Configuration | ✅ Complete | config.py |
| Data Loading | ✅ Complete | data_loader.py |
| Preprocessing | ✅ Complete | preprocessing.py |
| PPE Model | ✅ Complete | profiling/ppe.py |
| PREM Model | ⏳ To Create | profiling/prem.py |
| LBSM Model | ⏳ To Create | profiling/lbsm.py |
| LLPE Model | ⏳ To Create | profiling/llpe.py |
| IRT Model | ⏳ To Create | psychometric/irt_model.py |
| APDEA Algorithm | ⏳ To Create | algorithms/apdea.py |
| CTQS Algorithm | ⏳ To Create | algorithms/ctqs.py |
| BSSCA Algorithm | ⏳ To Create | algorithms/bssca.py |
| RAG Engine | ⏳ To Create | generation/rag_engine.py |
| Question Generator | ⏳ To Create | generation/question_generator.py |
| Bloom Classifier | ⏳ To Create | generation/bloom_classifier.py |
| Evaluation Metrics | ⏳ To Create | evaluation/metrics.py |
| Experiments | ⏳ To Create | experiments/run_experiments.py |

## 📦 Next Steps for Complete Implementation

1. **Create remaining profiling models** (PREM, LBSM, LLPE)
2. **Implement IRT psychometric model**
3. **Build core algorithms** (APDEA, CTQS, BSSCA)
4. **Develop RAG and question generation**
5. **Implement evaluation metrics**
6. **Create experiment runners**
7. **Add comprehensive tests**
8. **Generate documentation**

## 🤝 Contributing

Each file should include:
- Comprehensive docstrings
- Type hints
- Logging statements
- Unit tests
- Example usage

## 📚 References

Paper: "A Pedagogy-Guided Transformer-RAG Framework for Adaptive Question Generation and Learner Profiling"

Dataset: EduGen-Dataset (1,053 student records)
