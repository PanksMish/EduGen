"""
Context-Aware Transformer-RAG Question Synthesis (CTQS)
Implements Algorithm 6 from the paper

Generates pedagogy-aligned, difficulty-calibrated questions using:
- RAG for factual grounding
- Bloom taxonomy alignment
- IRT-based psychometric filtering
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

from config import CTQS_CONFIG, BLOOM_CONFIG, QUESTION_FILTER_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    """
    Generated question with metadata.
    """
    question_text: str
    correct_answer: str
    distractors: List[str]
    bloom_level: int
    bloom_level_name: str
    difficulty: str
    topic: str
    rationale: str
    confidence: float
    source_context: str
    
    def to_dict(self) -> dict:
        return {
            'question': self.question_text,
            'correct_answer': self.correct_answer,
            'distractors': self.distractors,
            'bloom_level': self.bloom_level,
            'bloom_level_name': self.bloom_level_name,
            'difficulty': self.difficulty,
            'topic': self.topic,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'source_context': self.source_context[:200] + '...' if len(self.source_context) > 200 else self.source_context
        }


class ContextAwareQuestionSynthesis:
    """
    CTQS: Context-Aware Transformer-RAG Question Synthesis.
    
    Implements Algorithm 6 from the paper:
    1. Retrieve top-k documents from knowledge base
    2. Construct structured prompt with Bloom + difficulty constraints
    3. Generate candidate questions
    4. Apply psychometric filtering
    5. Apply factual consistency check
    6. Select best question
    """
    
    def __init__(self,
                 rag_engine=None,
                 llm_generator=None,
                 bloom_classifier=None,
                 irt_model=None,
                 config: Optional[dict] = None):
        """
        Initialize CTQS.
        
        Args:
            rag_engine: Retrieval-Augmented Generation engine
            llm_generator: LLM for question generation
            bloom_classifier: Bloom taxonomy classifier
            irt_model: IRT model for difficulty estimation
            config: Configuration dictionary
        """
        if config is None:
            config = CTQS_CONFIG
        
        self.config = config
        self.rag_engine = rag_engine
        self.llm_generator = llm_generator
        self.bloom_classifier = bloom_classifier
        self.irt_model = irt_model
        
        logger.info("CTQS initialized")
        logger.info(f"  Max retries: {config['max_retries']}")
        logger.info(f"  Psychometric filtering: {config['psychometric_filtering']}")
        logger.info(f"  Factual checking: {config['factual_checking']}")
    
    def generate_question(self,
                         learner_state,
                         topic: str,
                         knowledge_base_path: Optional[str] = None) -> GeneratedQuestion:
        """
        Generate a single question adapted to learner state.
        
        Implements Algorithm 6 (CTQS).
        
        Args:
            learner_state: LearnerState from APDEA
            topic: Subject topic for the question
            knowledge_base_path: Path to knowledge base (optional)
            
        Returns:
            GeneratedQuestion object
        """
        logger.debug(f"Generating question for {learner_state}")
        
        # Extract learner parameters
        theta = learner_state.theta
        bloom_level = learner_state.bloom_level
        difficulty_band = learner_state.difficulty_band
        
        # Step 1: Retrieve relevant documents (RAG)
        retrieved_docs = self._retrieve_documents(topic, bloom_level, difficulty_band)
        
        # Step 2: Construct structured prompt
        prompt = self._construct_prompt(
            topic=topic,
            bloom_level=bloom_level,
            difficulty=difficulty_band,
            context=retrieved_docs
        )
        
        # Step 3: Generate candidate questions
        candidates = self._generate_candidates(prompt, n_candidates=5)
        
        # Step 4: Apply psychometric filtering
        if self.config['psychometric_filtering'] and self.irt_model is not None:
            candidates = self._filter_psychometric(candidates, theta)
        
        # Step 5: Apply factual consistency check
        if self.config['factual_checking']:
            candidates = self._filter_factual_consistency(candidates, retrieved_docs)
        
        # Step 6: Select best question
        if len(candidates) == 0:
            # Fallback: generate simple question
            logger.warning("No candidates passed filtering. Using fallback.")
            return self._generate_fallback_question(topic, bloom_level, difficulty_band)
        
        best_question = self._select_best_question(candidates, learner_state)
        
        logger.debug(f"Generated question: {best_question.question_text[:50]}...")
        
        return best_question
    
    def _retrieve_documents(self, 
                           topic: str,
                           bloom_level: int,
                           difficulty: str) -> List[Dict]:
        """
        Retrieve relevant documents from knowledge base.
        
        Corresponds to line 1 of Algorithm 6: D ← RAG(K, B*(i), d_i)
        
        Args:
            topic: Subject topic
            bloom_level: Target Bloom level
            difficulty: Target difficulty
            
        Returns:
            List of retrieved documents
        """
        if self.rag_engine is not None:
            # Use actual RAG engine
            docs = self.rag_engine.retrieve(
                query=topic,
                bloom_level=bloom_level,
                difficulty=difficulty,
                top_k=5
            )
            return docs
        else:
            # Fallback: return mock documents
            return [
                {
                    'content': f"This is relevant content about {topic}. "
                              f"It covers fundamental concepts and applications.",
                    'topic': topic,
                    'bloom_level': bloom_level,
                    'difficulty': difficulty,
                    'score': 0.9
                }
            ]
    
    def _construct_prompt(self,
                         topic: str,
                         bloom_level: int,
                         difficulty: str,
                         context: List[Dict]) -> str:
        """
        Construct structured prompt for question generation.
        
        Corresponds to line 2 of Algorithm 6.
        
        Args:
            topic: Subject topic
            bloom_level: Target Bloom level (1-3)
            difficulty: Target difficulty band
            context: Retrieved documents
            
        Returns:
            Formatted prompt string
        """
        bloom_names = {1: 'Remember', 2: 'Apply', 3: 'Analyze'}
        bloom_name = bloom_names[bloom_level]
        
        # Bloom-specific instructions
        bloom_instructions = {
            1: "Create a question that tests recall and recognition of facts, terms, or concepts.",
            2: "Create a question that requires applying knowledge to solve a problem or complete a task.",
            3: "Create a question that requires analyzing relationships, comparing concepts, or examining components."
        }
        
        # Combine context
        combined_context = "\n\n".join([doc['content'] for doc in context[:3]])
        
        prompt = f"""Generate a multiple-choice question based on the following context:

CONTEXT:
{combined_context}

REQUIREMENTS:
- Topic: {topic}
- Cognitive Level: {bloom_name} (Bloom's Taxonomy Level {bloom_level})
- Difficulty: {difficulty}
- Instruction: {bloom_instructions[bloom_level]}

CONSTRAINTS:
- Question must be clear and unambiguous
- Must have exactly 4 options (1 correct, 3 distractors)
- Distractors must be plausible but clearly incorrect
- Must be factually accurate based on the context provided
- Must align with the specified Bloom cognitive level

FORMAT YOUR RESPONSE AS JSON:
{{
    "question": "Your question text here?",
    "correct_answer": "The correct answer",
    "distractors": ["Distractor 1", "Distractor 2", "Distractor 3"],
    "rationale": "Brief explanation of why the correct answer is right"
}}"""
        
        return prompt
    
    def _generate_candidates(self, prompt: str, n_candidates: int = 5) -> List[Dict]:
        """
        Generate multiple candidate questions.
        
        Corresponds to line 3 of Algorithm 6: Q = {q1, ..., qm}
        
        Args:
            prompt: Formatted prompt
            n_candidates: Number of candidates to generate
            
        Returns:
            List of candidate question dictionaries
        """
        candidates = []
        
        if self.llm_generator is not None:
            # Use actual LLM
            for _ in range(n_candidates):
                try:
                    response = self.llm_generator.generate(prompt)
                    candidate = json.loads(response)
                    candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Failed to generate candidate: {e}")
                    continue
        else:
            # Fallback: generate mock candidates
            for i in range(n_candidates):
                candidates.append({
                    'question': f"What is the main concept related to this topic? (Candidate {i+1})",
                    'correct_answer': "The correct concept",
                    'distractors': [
                        "Incorrect concept A",
                        "Incorrect concept B",
                        "Incorrect concept C"
                    ],
                    'rationale': "This is the correct answer because it accurately describes the concept."
                })
        
        return candidates
    
    def _filter_psychometric(self, 
                            candidates: List[Dict],
                            theta: float) -> List[Dict]:
        """
        Filter candidates based on psychometric compatibility.
        
        Corresponds to lines 4-8 of Algorithm 6.
        
        Args:
            candidates: List of candidate questions
            theta: Learner ability estimate
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for candidate in candidates:
            # Estimate item difficulty (b parameter)
            # In practice, this would use IRT calibration
            # For now, approximate based on question complexity
            question_length = len(candidate['question'])
            estimated_difficulty = (question_length - 50) / 100  # Rough approximation
            
            # Compute compatibility: |θ - b|
            compatibility = abs(theta - estimated_difficulty)
            
            # Accept if within threshold
            if compatibility <= self.config['compatibility_threshold']:
                candidate['compatibility'] = compatibility
                candidate['estimated_difficulty'] = estimated_difficulty
                filtered.append(candidate)
        
        logger.debug(f"Psychometric filter: {len(candidates)} → {len(filtered)} candidates")
        
        return filtered
    
    def _filter_factual_consistency(self,
                                   candidates: List[Dict],
                                   context_docs: List[Dict]) -> List[Dict]:
        """
        Check factual consistency with retrieved context.
        
        Corresponds to lines 9-13 of Algorithm 6.
        
        Args:
            candidates: List of candidate questions
            context_docs: Retrieved context documents
            
        Returns:
            Filtered list of factually consistent candidates
        """
        filtered = []
        combined_context = " ".join([doc['content'] for doc in context_docs])
        
        for candidate in candidates:
            # Check if answer appears in or is supported by context
            # Simplified check: answer text present in context
            answer = candidate['correct_answer'].lower()
            context_lower = combined_context.lower()
            
            # Simple keyword matching (in practice, use semantic similarity)
            keywords = answer.split()
            support_score = sum(1 for kw in keywords if kw in context_lower) / max(len(keywords), 1)
            
            # Accept if sufficiently supported
            if support_score >= QUESTION_FILTER_CONFIG['consistency_threshold']:
                candidate['factual_support'] = support_score
                filtered.append(candidate)
        
        logger.debug(f"Factual filter: {len(candidates)} → {len(filtered)} candidates")
        
        return filtered
    
    def _select_best_question(self,
                             candidates: List[Dict],
                             learner_state) -> GeneratedQuestion:
        """
        Select the best question from candidates.
        
        Corresponds to lines 14-15 of Algorithm 6: q* = argmin_qj Cj
        
        Args:
            candidates: Filtered candidate questions
            learner_state: Learner state from APDEA
            
        Returns:
            Best GeneratedQuestion object
        """
        if len(candidates) == 0:
            raise ValueError("No candidates available")
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            score = 0.0
            
            # Psychometric compatibility (lower is better)
            if 'compatibility' in candidate:
                score -= candidate['compatibility'] * 2.0
            
            # Factual support (higher is better)
            if 'factual_support' in candidate:
                score += candidate['factual_support'] * 1.5
            
            # Question quality (length, clarity)
            q_len = len(candidate['question'])
            if 30 < q_len < 200:  # Reasonable length
                score += 1.0
            
            candidate['final_score'] = score
            scored_candidates.append((score, candidate))
        
        # Select highest scoring
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best = scored_candidates[0][1]
        
        # Create GeneratedQuestion object
        bloom_names = {1: 'Remember', 2: 'Apply', 3: 'Analyze'}
        
        question = GeneratedQuestion(
            question_text=best['question'],
            correct_answer=best['correct_answer'],
            distractors=best['distractors'],
            bloom_level=learner_state.bloom_level,
            bloom_level_name=bloom_names[learner_state.bloom_level],
            difficulty=learner_state.difficulty_band,
            topic="General",  # Would be extracted from context
            rationale=best.get('rationale', 'No rationale provided'),
            confidence=best.get('final_score', 0.5),
            source_context="Context from retrieved documents"
        )
        
        return question
    
    def _generate_fallback_question(self,
                                   topic: str,
                                   bloom_level: int,
                                   difficulty: str) -> GeneratedQuestion:
        """
        Generate a simple fallback question when all candidates fail.
        
        Args:
            topic: Subject topic
            bloom_level: Target Bloom level
            difficulty: Target difficulty
            
        Returns:
            Fallback GeneratedQuestion
        """
        bloom_names = {1: 'Remember', 2: 'Apply', 3: 'Analyze'}
        
        question = GeneratedQuestion(
            question_text=f"What is a key concept in {topic}?",
            correct_answer=f"A fundamental concept in {topic}",
            distractors=[
                "An unrelated concept",
                "An incorrect interpretation",
                "A misunderstood principle"
            ],
            bloom_level=bloom_level,
            bloom_level_name=bloom_names[bloom_level],
            difficulty=difficulty,
            topic=topic,
            rationale="This fallback question covers basic understanding.",
            confidence=0.3,
            source_context="Fallback generation"
        )
        
        return question
    
    def generate_quiz(self,
                     learner_state,
                     topic: str,
                     n_questions: int = 10) -> List[GeneratedQuestion]:
        """
        Generate a complete adaptive quiz.
        
        Args:
            learner_state: LearnerState from APDEA
            topic: Subject topic
            n_questions: Number of questions to generate
            
        Returns:
            List of GeneratedQuestion objects
        """
        logger.info(f"Generating adaptive quiz: {n_questions} questions on {topic}")
        
        questions = []
        
        for i in range(n_questions):
            try:
                question = self.generate_question(learner_state, topic)
                questions.append(question)
                logger.debug(f"Generated question {i+1}/{n_questions}")
            except Exception as e:
                logger.error(f"Failed to generate question {i+1}: {e}")
                continue
        
        logger.info(f"Quiz generation complete: {len(questions)} questions")
        
        return questions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    from algorithms.apdea import LearnerState
    
    print("=" * 80)
    print("CTQS (CONTEXT-AWARE QUESTION SYNTHESIS) DEMO")
    print("=" * 80)
    
    # Create a mock learner state
    learner_state = LearnerState(
        theta=0.5,
        bloom_level=2,
        bloom_level_name='Apply',
        difficulty_band='Medium',
        confidence=0.85,
        profiling_scores={}
    )
    
    print(f"\nLearner State: {learner_state}")
    
    # Initialize CTQS (without actual models for demo)
    ctqs = ContextAwareQuestionSynthesis()
    
    # Generate a single question
    print("\n" + "=" * 80)
    print("GENERATING SINGLE QUESTION")
    print("=" * 80)
    
    question = ctqs.generate_question(
        learner_state=learner_state,
        topic="Linear Algebra"
    )
    
    print("\nGenerated Question:")
    print(json.dumps(question.to_dict(), indent=2))
    
    # Generate a quiz
    print("\n" + "=" * 80)
    print("GENERATING ADAPTIVE QUIZ")
    print("=" * 80)
    
    quiz = ctqs.generate_quiz(
        learner_state=learner_state,
        topic="Calculus",
        n_questions=5
    )
    
    print(f"\nGenerated {len(quiz)} questions")
    for i, q in enumerate(quiz, 1):
        print(f"\n{i}. {q.question_text}")
        print(f"   Bloom: {q.bloom_level_name}, Difficulty: {q.difficulty}")
    
    print("\n" + "=" * 80)
    print("CTQS Demo Complete!")
    print("=" * 80)
