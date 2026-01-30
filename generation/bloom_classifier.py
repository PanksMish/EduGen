"""
Bloom's Taxonomy Classifier
Classifies questions into Bloom cognitive levels
"""

import numpy as np
import logging
from typing import Optional, Dict, List
import re

from config import BLOOM_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BloomClassifier:
    """
    Classifier for Bloom's Taxonomy cognitive levels.
    
    Uses keyword matching and pattern recognition to classify
    questions into Bloom levels: Remember, Apply, Analyze
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Bloom classifier.
        
        Args:
            config: Configuration dictionary. Uses BLOOM_CONFIG if None.
        """
        if config is None:
            config = BLOOM_CONFIG
        
        self.config = config
        self.keywords = config['keywords']
        self.levels = config['levels']
        
        # Compile patterns for efficiency
        self.patterns = self._compile_patterns()
        
        logger.info("Bloom Classifier initialized")
        logger.info(f"  Active levels: {config['active_levels']}")
    
    def _compile_patterns(self) -> Dict[int, List[re.Pattern]]:
        """
        Compile regex patterns for each Bloom level.
        
        Returns:
            Dictionary mapping levels to compiled patterns
        """
        patterns = {}
        
        for level, keywords in self.keywords.items():
            patterns[level] = [
                re.compile(r'\b' + keyword + r'\b', re.IGNORECASE)
                for keyword in keywords
            ]
        
        return patterns
    
    def classify(self, question_text: str) -> Dict:
        """
        Classify a question into Bloom cognitive level.
        
        Args:
            question_text: Question text to classify
            
        Returns:
            Dictionary with classification results
        """
        # Count keyword matches for each level
        level_scores = {}
        
        for level, patterns in self.patterns.items():
            score = 0
            matched_keywords = []
            
            for pattern in patterns:
                if pattern.search(question_text):
                    score += 1
                    matched_keywords.append(pattern.pattern.strip(r'\b'))
            
            level_scores[level] = {
                'score': score,
                'keywords': matched_keywords
            }
        
        # Determine predicted level
        if all(s['score'] == 0 for s in level_scores.values()):
            # No keywords matched, use heuristics
            predicted_level = self._classify_by_heuristics(question_text)
        else:
            # Level with highest score
            predicted_level = max(
                level_scores.keys(),
                key=lambda k: level_scores[k]['score']
            )
        
        # Get confidence
        total_score = sum(s['score'] for s in level_scores.values())
        confidence = (
            level_scores[predicted_level]['score'] / total_score
            if total_score > 0 else 0.5
        )
        
        result = {
            'predicted_level': predicted_level,
            'predicted_level_name': self.levels[predicted_level],
            'confidence': float(confidence),
            'level_scores': {
                self.levels[k]: v['score']
                for k, v in level_scores.items()
            },
            'matched_keywords': level_scores[predicted_level]['keywords']
        }
        
        return result
    
    def _classify_by_heuristics(self, question_text: str) -> int:
        """
        Classify using heuristic rules when no keywords match.
        
        Args:
            question_text: Question text
            
        Returns:
            Predicted Bloom level
        """
        text_lower = question_text.lower()
        
        # Heuristic 1: Question type
        if any(text_lower.startswith(w) for w in ['what is', 'who is', 'when did', 'where is']):
            return 1  # Remember
        
        # Heuristic 2: Complexity indicators
        if any(word in text_lower for word in ['calculate', 'compute', 'solve', 'find']):
            return 2  # Apply
        
        if any(word in text_lower for word in ['compare', 'contrast', 'analyze', 'explain why']):
            return 3  # Analyze
        
        # Default to Remember
        return 1
    
    def classify_batch(self, questions: List[str]) -> List[Dict]:
        """
        Classify multiple questions.
        
        Args:
            questions: List of question texts
            
        Returns:
            List of classification results
        """
        return [self.classify(q) for q in questions]
    
    def get_bloom_distribution(self, questions: List[str]) -> Dict:
        """
        Get distribution of Bloom levels in a set of questions.
        
        Args:
            questions: List of question texts
            
        Returns:
            Dictionary with distribution statistics
        """
        classifications = self.classify_batch(questions)
        
        distribution = {
            level_name: 0
            for level_name in self.levels.values()
        }
        
        for result in classifications:
            distribution[result['predicted_level_name']] += 1
        
        # Convert to percentages
        total = len(questions)
        distribution_pct = {
            level: (count / total * 100)
            for level, count in distribution.items()
        }
        
        return {
            'counts': distribution,
            'percentages': distribution_pct,
            'total_questions': total
        }
    
    def validate_alignment(self,
                          question_text: str,
                          target_level: int,
                          threshold: float = 0.6) -> bool:
        """
        Validate if a question aligns with target Bloom level.
        
        Args:
            question_text: Question text
            target_level: Target Bloom level (1-3)
            threshold: Minimum confidence required
            
        Returns:
            True if aligned, False otherwise
        """
        result = self.classify(question_text)
        
        aligned = (
            result['predicted_level'] == target_level and
            result['confidence'] >= threshold
        )
        
        return aligned
    
    def suggest_improvements(self,
                           question_text: str,
                           target_level: int) -> Dict:
        """
        Suggest improvements to align question with target level.
        
        Args:
            question_text: Question text
            target_level: Target Bloom level
            
        Returns:
            Dictionary with suggestions
        """
        current = self.classify(question_text)
        current_level = current['predicted_level']
        
        suggestions = {
            'current_level': self.levels[current_level],
            'target_level': self.levels[target_level],
            'aligned': current_level == target_level,
            'suggestions': []
        }
        
        if current_level == target_level:
            suggestions['suggestions'].append("Question is already well-aligned!")
            return suggestions
        
        # Generate suggestions based on target level
        target_keywords = self.keywords[target_level]
        
        if target_level == 1:  # Remember
            suggestions['suggestions'] = [
                f"Use recall verbs like: {', '.join(target_keywords[:3])}",
                "Ask for definitions or facts",
                "Focus on 'what', 'who', 'when' questions"
            ]
        elif target_level == 2:  # Apply
            suggestions['suggestions'] = [
                f"Use application verbs like: {', '.join(target_keywords[:3])}",
                "Ask learner to solve a problem",
                "Require using knowledge in new situations"
            ]
        elif target_level == 3:  # Analyze
            suggestions['suggestions'] = [
                f"Use analysis verbs like: {', '.join(target_keywords[:3])}",
                "Ask for comparisons or relationships",
                "Require breaking down concepts"
            ]
        
        return suggestions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("BLOOM'S TAXONOMY CLASSIFIER DEMO")
    print("=" * 80)
    
    # Initialize classifier
    classifier = BloomClassifier()
    
    # Test questions
    test_questions = [
        "What is the definition of calculus?",
        "Calculate the derivative of f(x) = x^2 + 3x + 2",
        "Compare and contrast differential and integral calculus",
        "List the main branches of mathematics",
        "Solve the equation 2x + 5 = 15",
        "Analyze why the limit of 1/x approaches infinity as x approaches zero"
    ]
    
    print("\n" + "=" * 80)
    print("CLASSIFYING INDIVIDUAL QUESTIONS")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        result = classifier.classify(question)
        
        print(f"\n{i}. {question}")
        print(f"   Level: {result['predicted_level_name']} (confidence: {result['confidence']:.2f})")
        if result['matched_keywords']:
            print(f"   Keywords: {', '.join(result['matched_keywords'])}")
    
    # Get distribution
    print("\n" + "=" * 80)
    print("BLOOM LEVEL DISTRIBUTION")
    print("=" * 80)
    
    distribution = classifier.get_bloom_distribution(test_questions)
    
    print(f"\nTotal questions: {distribution['total_questions']}")
    print("\nDistribution:")
    for level, count in distribution['counts'].items():
        pct = distribution['percentages'][level]
        print(f"  {level}: {count} ({pct:.1f}%)")
    
    # Test alignment validation
    print("\n" + "=" * 80)
    print("ALIGNMENT VALIDATION")
    print("=" * 80)
    
    test_cases = [
        (test_questions[0], 1, "Remember"),
        (test_questions[1], 2, "Apply"),
        (test_questions[2], 3, "Analyze")
    ]
    
    for question, target_level, target_name in test_cases:
        aligned = classifier.validate_alignment(question, target_level)
        print(f"\nQuestion: {question[:50]}...")
        print(f"Target: {target_name}")
        print(f"Aligned: {'✓' if aligned else '✗'}")
    
    # Test suggestions
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 80)
    
    question = "What is a matrix?"
    target = 2  # Apply
    
    suggestions = classifier.suggest_improvements(question, target)
    
    print(f"\nQuestion: {question}")
    print(f"Current level: {suggestions['current_level']}")
    print(f"Target level: {suggestions['target_level']}")
    print(f"Aligned: {suggestions['aligned']}")
    print("\nSuggestions:")
    for s in suggestions['suggestions']:
        print(f"  - {s}")
    
    print("\n" + "=" * 80)
    print("Bloom Classifier Demo Complete!")
    print("=" * 80)
