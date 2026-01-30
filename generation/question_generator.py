"""
Question Generator
Interfaces with LLMs/SLMs for question generation
Supports multiple backends: OpenAI, Anthropic, local models
"""

import numpy as np
import logging
from typing import Optional, Dict, List
import json
import time

from config import LLM_QG_CONFIG, SLM_QG_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available")


class QuestionGenerator:
    """
    Question generator with LLM/SLM support.
    
    Supports multiple backends:
    - OpenAI (GPT-3.5, GPT-4)
    - Anthropic (Claude)
    - Local models (via Hugging Face)
    - Mock mode (for testing without API keys)
    """
    
    def __init__(self,
                 model_type: str = 'mock',
                 model_size: str = 'large',
                 api_key: Optional[str] = None,
                 config: Optional[dict] = None):
        """
        Initialize question generator.
        
        Args:
            model_type: Type of model ('openai', 'anthropic', 'local', 'mock')
            model_size: 'large' (LLM) or 'small' (SLM)
            api_key: API key for commercial services
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.model_size = model_size
        self.api_key = api_key
        
        # Select config based on size
        if config is None:
            config = LLM_QG_CONFIG if model_size == 'large' else SLM_QG_CONFIG
        
        self.config = config
        self.client = None
        
        # Initialize client
        self._initialize_client()
        
        logger.info(f"Question Generator initialized")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Model size: {model_size}")
        logger.info(f"  Model name: {config['model_name']}")
    
    def _initialize_client(self):
        """Initialize the API client based on model type"""
        if self.model_type == 'openai':
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI library not installed")
                self.model_type = 'mock'
                return
            
            if self.api_key:
                openai.api_key = self.api_key
            
            self.client = openai
            logger.info("OpenAI client initialized")
        
        elif self.model_type == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                logger.error("Anthropic library not installed")
                self.model_type = 'mock'
                return
            
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            
            logger.info("Anthropic client initialized")
        
        elif self.model_type == 'local':
            logger.info("Local model mode (requires manual setup)")
            # Would load local model here
        
        else:  # mock
            logger.info("Mock mode (for testing without API)")
            self.client = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the configured model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Override config with kwargs
        temperature = kwargs.get('temperature', self.config['temperature'])
        max_tokens = kwargs.get('max_tokens', self.config['max_tokens'])
        
        try:
            if self.model_type == 'openai' and OPENAI_AVAILABLE:
                return self._generate_openai(prompt, temperature, max_tokens)
            
            elif self.model_type == 'anthropic' and ANTHROPIC_AVAILABLE:
                return self._generate_anthropic(prompt, temperature, max_tokens)
            
            elif self.model_type == 'local':
                return self._generate_local(prompt, temperature, max_tokens)
            
            else:  # mock
                return self._generate_mock(prompt)
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._generate_mock(prompt)
    
    def _generate_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using OpenAI API"""
        response = openai.ChatCompletion.create(
            model=self.config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using Anthropic API"""
        message = self.client.messages.create(
            model=self.config['model_name'],
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def _generate_local(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using local model"""
        # Placeholder for local model generation
        logger.warning("Local model generation not implemented. Using mock.")
        return self._generate_mock(prompt)
    
    def _generate_mock(self, prompt: str) -> str:
        """Generate mock response for testing"""
        # Extract topic and level from prompt if possible
        import re
        
        topic_match = re.search(r'Topic: (.+)', prompt)
        level_match = re.search(r'Level (\d)', prompt)
        
        topic = topic_match.group(1) if topic_match else "the subject"
        level = level_match.group(1) if level_match else "2"
        
        # Generate mock question in JSON format
        mock_response = {
            "question": f"What is a key concept in {topic}?",
            "correct_answer": f"The fundamental principle of {topic}",
            "distractors": [
                f"An incorrect interpretation of {topic}",
                f"A misunderstood aspect of {topic}",
                f"An unrelated concept to {topic}"
            ],
            "rationale": f"This tests understanding of {topic} at cognitive level {level}."
        }
        
        return json.dumps(mock_response, indent=2)
    
    def generate_question(self,
                         topic: str,
                         bloom_level: int,
                         difficulty: str,
                         context: Optional[str] = None) -> Dict:
        """
        Generate a complete question with metadata.
        
        Args:
            topic: Subject topic
            bloom_level: Bloom cognitive level (1-3)
            difficulty: Difficulty level ('Easy', 'Medium', 'Hard')
            context: Optional context from RAG
            
        Returns:
            Dictionary with question data
        """
        # Construct prompt
        bloom_names = {1: 'Remember', 2: 'Apply', 3: 'Analyze'}
        bloom_name = bloom_names[bloom_level]
        
        prompt = f"""Generate a multiple-choice question based on the following:

TOPIC: {topic}
COGNITIVE LEVEL: {bloom_name} (Bloom's Taxonomy Level {bloom_level})
DIFFICULTY: {difficulty}
"""
        
        if context:
            prompt += f"\nCONTEXT:\n{context}\n"
        
        prompt += """
REQUIREMENTS:
- Question must be clear and unambiguous
- Must have exactly 4 options (1 correct, 3 distractors)
- Distractors must be plausible but incorrect
- Must align with the specified Bloom cognitive level
- Must be factually accurate

FORMAT YOUR RESPONSE AS JSON:
{
    "question": "Your question text here?",
    "correct_answer": "The correct answer",
    "distractors": ["Distractor 1", "Distractor 2", "Distractor 3"],
    "rationale": "Brief explanation of why the correct answer is right"
}
"""
        
        # Generate
        response = self.generate(prompt)
        
        # Parse JSON
        try:
            question_data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                question_data = json.loads(json_match.group())
            else:
                # Fallback
                question_data = {
                    'question': f"What is a key concept in {topic}?",
                    'correct_answer': "The main concept",
                    'distractors': ["Wrong 1", "Wrong 2", "Wrong 3"],
                    'rationale': "Testing basic understanding"
                }
        
        # Add metadata
        question_data['topic'] = topic
        question_data['bloom_level'] = bloom_level
        question_data['bloom_level_name'] = bloom_name
        question_data['difficulty'] = difficulty
        
        return question_data
    
    def generate_batch(self,
                      specifications: List[Dict],
                      delay: float = 0.5) -> List[Dict]:
        """
        Generate multiple questions in batch.
        
        Args:
            specifications: List of question specifications
            delay: Delay between API calls (seconds)
            
        Returns:
            List of generated questions
        """
        questions = []
        
        for i, spec in enumerate(specifications):
            logger.info(f"Generating question {i+1}/{len(specifications)}")
            
            question = self.generate_question(
                topic=spec.get('topic', 'General'),
                bloom_level=spec.get('bloom_level', 2),
                difficulty=spec.get('difficulty', 'Medium'),
                context=spec.get('context')
            )
            
            questions.append(question)
            
            # Rate limiting
            if i < len(specifications) - 1 and delay > 0:
                time.sleep(delay)
        
        return questions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("QUESTION GENERATOR DEMO")
    print("=" * 80)
    
    # Initialize generator in mock mode
    generator = QuestionGenerator(model_type='mock', model_size='large')
    
    # Test single question generation
    print("\n" + "=" * 80)
    print("GENERATING SINGLE QUESTION")
    print("=" * 80)
    
    question = generator.generate_question(
        topic="Calculus",
        bloom_level=2,
        difficulty="Medium",
        context="Calculus is the study of continuous change."
    )
    
    print("\nGenerated Question:")
    print(json.dumps(question, indent=2))
    
    # Test batch generation
    print("\n" + "=" * 80)
    print("GENERATING BATCH OF QUESTIONS")
    print("=" * 80)
    
    specifications = [
        {'topic': 'Linear Algebra', 'bloom_level': 1, 'difficulty': 'Easy'},
        {'topic': 'Probability', 'bloom_level': 2, 'difficulty': 'Medium'},
        {'topic': 'Statistics', 'bloom_level': 3, 'difficulty': 'Hard'},
    ]
    
    questions = generator.generate_batch(specifications, delay=0.1)
    
    print(f"\nGenerated {len(questions)} questions")
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. {q['question']}")
        print(f"   Level: {q['bloom_level_name']}, Difficulty: {q['difficulty']}")
    
    print("\n" + "=" * 80)
    print("Question Generator Demo Complete!")
    print("=" * 80)
    
    print("\nNOTE: This demo uses mock generation.")
    print("To use real LLMs, set model_type='openai' or 'anthropic'")
    print("and provide API key.")
