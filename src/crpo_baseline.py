from groq import Groq
import json
from typing import List, Dict, Tuple, Any, Optional
import time
from src.config import GROQ_API_KEY, DEFAULT_MODEL, TEMPERATURE

class CRPOBaseline:
    """Contrastive Reasoning Prompt Optimization - Single Domain"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("API Key must be provided or set in environment variables.")
        self.client = Groq(api_key=self.api_key)
        self.api_calls = 0
    
    def retrieve_reference_examples(self, 
                                   helpsteer2_data: List[Dict[str, Any]],
                                   domain: str,
                                   k: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve k high-quality and k low-quality examples"""
        
        # Sort by quality score
        sorted_data = sorted(helpsteer2_data, 
                           key=lambda x: x.get('quality_score', 0),
                           reverse=True)
        
        # High quality: top k
        high_quality = sorted_data[:k]
        
        # Low quality: bottom k
        low_quality = sorted_data[-k:]
        
        return high_quality, low_quality
    
    def tiered_contrastive_reasoning(self,
                                    task: str,
                                    high_examples: List[Dict[str, Any]],
                                    low_examples: List[Dict[str, Any]]) -> str:
        """Ask LLM to reason about why high > low quality"""
        
        # Format examples for display
        high_text = "\n".join([
            f"Example {i+1}: {e.get('prompt', '')[:100]}... → Response quality: HIGH"
            for i, e in enumerate(high_examples[:3])  # Show top 3
        ])
        
        low_text = "\n".join([
            f"Example {i+1}: {e.get('prompt', '')[:100]}... → Response quality: LOW"
            for i, e in enumerate(low_examples[:3])
        ])
        
        reasoning_prompt = f"""
You are a prompt optimization expert. Analyze these high vs low quality examples.

HIGH QUALITY EXAMPLES (Good responses):
{high_text}

LOW QUALITY EXAMPLES (Poor responses):
{low_text}

For this task: '{task}'

Question: What are the key properties that make prompts effective for this task? 
Consider:
1. How should the prompt be structured?
2. What kind of reasoning should be encouraged?
3. What specific instructions lead to better responses?

Please provide 3-4 key insights about what makes prompts work better.
"""
        
        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": reasoning_prompt}],
            temperature=TEMPERATURE,
            max_tokens=300
        )
        
        self.api_calls += 1
        reasoning = response.choices[0].message.content
        
        return reasoning
    
    def generate_optimized_prompt(self, 
                             task: str,
                             reasoning: str) -> str:
        """Generate optimized prompt based on reasoning"""
        
        generation_prompt = f"""
Based on the following insights about effective prompts:

{reasoning}

Generate an optimized prompt for this task:
Task: {task}

IMPORTANT REQUIREMENTS:
1. The prompt should be GENERIC and work for ANY example in this domain
2. Do NOT include specific concrete examples in the prompt template
3. Do NOT include hardcoded problem descriptions
4. Use ONLY the placeholder {{question}} - do not use {{problem}}, {{statement}}, etc.
5. Focus on process and methodology, not specific content
6. Be clear and specific about the reasoning approach, not the problem content

Output ONLY the optimized prompt template, no explanation.
Example: "Solve the following math problem step-by-step: {{question}}"
"""
    
        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": generation_prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        self.api_calls += 1
        optimized_prompt = response.choices[0].message.content.strip()
        
        # Normalize placeholders to use {question}
        optimized_prompt = optimized_prompt.replace('{problem}', '{question}')
        optimized_prompt = optimized_prompt.replace('{statement}', '{question}')
        optimized_prompt = optimized_prompt.replace('{prompt}', '{question}')
        
        # Clean up quotes
        optimized_prompt = optimized_prompt.replace('"', '').strip()
        
        return optimized_prompt


    
    def optimize(self, 
                task: str,
                helpsteer2_data: List[Dict[str, Any]],
                domain: str = 'general') -> Dict[str, Any]:
        """Full single-domain CRPO pipeline"""
        
        print(f"\n" + "=" * 60)
        print(f"SINGLE-DOMAIN CRPO OPTIMIZATION")
        print(f"Task: {task}")
        print(f"Domain: {domain}")
        print("=" * 60)
        
        # Step 1: Retrieve examples
        print("\nStep 1: Retrieving reference examples...")
        high, low = self.retrieve_reference_examples(helpsteer2_data, domain, k=5)
        print(f"   Retrieved {len(high)} high-quality and {len(low)} low-quality examples")
        
        # Step 2: Tiered reasoning
        print("\nStep 2: Contrastive reasoning...")
        reasoning = self.tiered_contrastive_reasoning(task, high, low)
        print(f"   Generated reasoning (length: {len(reasoning)} chars)")
        print(f"   Sample: {reasoning[:200]}...")
        
        # Step 3: Generate prompt
        print("\nStep 3: Generating optimized prompt...")
        optimized_prompt = self.generate_optimized_prompt(task, reasoning)
        print(f"   Generated prompt length: {len(optimized_prompt)} chars")
        print(f"   Prompt:\n   {optimized_prompt[:300]}...")
        
        print("\n" + "=" * 60)
        print(f"OPTIMIZATION COMPLETE")
        print(f"Total API calls: {self.api_calls}")
        print("=" * 60)
        
        return {
            'task': task,
            'domain': domain,
            'reasoning': reasoning,
            'optimized_prompt': optimized_prompt,
            'api_calls': self.api_calls
        }
