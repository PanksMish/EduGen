"""
EduGen Algorithms Module
Core algorithms: APDEA, CTQS, BSSCA
"""

from .apdea import AdaptiveProfilingDifficultyEstimation
from .ctqs import ContextAwareQuestionSynthesis
from .bssca import BenchmarkScholarshipComputation

__all__ = [
    'AdaptiveProfilingDifficultyEstimation',
    'ContextAwareQuestionSynthesis',
    'BenchmarkScholarshipComputation',
]

__version__ = '1.0.0'
