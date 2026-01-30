# EduGen: Pedagogy-Guided Transformer-RAG Framework

A comprehensive adaptive quiz generation framework that integrates learner profiling, psychometric modelling, and retrieval-augmented Transformer-based generation for personalized educational assessment.

## 📋 Overview

EduGen is an advanced educational assessment system that:
- Generates pedagogy-aware, factually grounded quiz questions
- Profiles learners using multi-model analytics
- Adapts difficulty based on psychometric principles (IRT)
- Aligns questions with Bloom's Taxonomy cognitive levels
- Reduces hallucinations through RAG-based verification

## 🏗️ Architecture

The framework consists of three main components:

1. **Learner Profiling Engine**
   - Performance Probability Estimator (PPE)
   - Pedagogical Rule Extraction Model (PREM)
   - Learner Behaviour Segmentation Module (LBSM)
   - Latent Learning Pattern Extractor (LLPE)

2. **Adaptive Assessment Generation**
   - Context-Aware Transformer-RAG Question Synthesis (CTQS)
   - IRT-based difficulty calibration
   - Bloom's Taxonomy alignment

3. **Benchmarking & Analytics**
   - Benchmark and Scholarship Suitability Computation (BSSCA)
   - Performance tracking and reporting

## 📊 Dataset

The implementation uses the `EduGen-Dataset` with 1,053 student assessment records including:
- 12 quiz scores per student
- Attendance records
- Behavioral indicators
- Study habits
- Engineered features (Consistency Index, Top 9 Sum)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/edugen-framework.git
cd edugen-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python scripts/download_models.py
```

## 💻 Usage

### Basic Usage

```python
from edugen import EduGenFramework

# Initialize framework
edugen = EduGenFramework(
    use_llm=True,
    model_size='large',
    enable_rag=True
)

# Load student data
edugen.load_data('data/student_performance.csv')

# Train profiling models
edugen.train_profiling_models()

# Generate adaptive quiz
student_id = 'S001'
quiz = edugen.generate_adaptive_quiz(
    student_id=student_id,
    num_questions=10,
    subject='Mathematics'
)

# Evaluate and benchmark
results = edugen.evaluate_student(student_id)
print(f"Benchmark Score: {results['benchmark_score']:.2f}")
print(f"Scholarship Suitability: {results['scholarship_score']:.2f}")
```

### Advanced Usage

```python
# Custom profiling
learner_profile = edugen.profile_learner('S001')
print(f"Ability: {learner_profile['theta']:.2f}")
print(f"Recommended Level: {learner_profile['bloom_level']}")

# Generate questions with specific constraints
questions = edugen.generate_questions(
    topic='Calculus',
    bloom_level='Apply',
    difficulty='Medium',
    num_questions=5
)

# Batch processing
batch_results = edugen.batch_evaluate(student_ids=['S001', 'S002', 'S003'])
```

## 📁 Project Structure

```
edugen-framework/
├── data/
│   ├── raw/                    # Raw student performance data
│   ├── processed/              # Preprocessed datasets
│   └── knowledge_base/         # Syllabus content for RAG
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_loader.py         # Dataset loading utilities
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── ppe.py            # Performance Probability Estimator
│   │   ├── prem.py           # Pedagogical Rule Extraction
│   │   ├── lbsm.py           # Behaviour Segmentation
│   │   └── llpe.py           # Latent Pattern Extractor
│   ├── psychometric/
│   │   ├── __init__.py
│   │   └── irt_model.py      # Item Response Theory
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── rag_engine.py     # Retrieval-Augmented Generation
│   │   ├── question_gen.py   # Question generation
│   │   └── bloom_classifier.py # Bloom's Taxonomy alignment
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── apdea.py          # Adaptive Profiling & Difficulty Estimation
│   │   ├── ctqs.py           # Context-Aware Question Synthesis
│   │   └── bssca.py          # Benchmark & Scholarship Computation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py        # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── helpers.py        # Utility functions
├── experiments/
│   ├── run_experiments.py    # Main experiment runner
│   ├── ablation_study.py     # Ablation experiments
│   └── pilot_study.py        # Empirical pilot study
├── models/                    # Saved model weights
├── results/                   # Experimental results
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Unit tests
├── scripts/
│   └── download_models.py    # Model download script
├── requirements.txt
├── setup.py
└── README.md
```

## 🔬 Experiments

Run the complete experimental suite:

```bash
# Run main experiments
python experiments/run_experiments.py

# Run ablation study
python experiments/ablation_study.py --config configs/ablation_config.yaml

# Run pilot study analysis
python experiments/pilot_study.py --group experimental
```

## 📈 Results

The framework achieves:
- **91.2% Pedagogical Success Rate** (LLM-QG)
- **87.5% PSR** (SLM-QG - efficient variant)
- **89.4% F1-Score** with low hallucination rate
- Strong performance across all Bloom's Taxonomy levels
- Excellent calibration (ECE: 0.073)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@article{edugen2026,
  title={EduGen: A Pedagogy-Guided Transformer-RAG Framework for Adaptive Question Generation and Learner Profiling},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## 📧 Contact

For questions and support:
- Email: pankaj.mishra@somaiya.edu

## 🙏 Acknowledgments

This work builds upon research in:
- Transformer architectures and LLMs
- Item Response Theory (IRT)
- Bloom's Taxonomy
- Retrieval-Augmented Generation (RAG)
