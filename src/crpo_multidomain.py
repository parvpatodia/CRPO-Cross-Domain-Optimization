from groq import Groq
from typing import List, Dict, Any, Optional
import json
from src.config import GROQ_API_KEY, DEFAULT_MODEL, TEMPERATURE

class CRPOMultiDomain:
    """Contrastive Reasoning Prompt Optimization - Multi-Domain Variant"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("API Key must be provided or set in environment variables.")
        self.client = Groq(api_key=self.api_key)
        self.api_calls = 0
    
    def retrieve_multidomain_examples(self,
                                     helpsteer2_data: List[Dict[str, Any]],
                                     k: int = 3) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Retrieve k examples from EACH domain (simulated by quality tiers)."""
        
        # Sort by quality score
        sorted_data = sorted(helpsteer2_data,
                           key=lambda x: x.get('quality_score', 0),
                           reverse=True)
        
        # Simulate 4 domains by splitting data into quality tiers
        chunk_size = len(sorted_data) // 4
        
        multidomain = {
            'math': {
                'high': sorted_data[:k],
                'low': sorted_data[-k:]
            },
            'reasoning': {
                'high': sorted_data[chunk_size:chunk_size+k],
                'low': sorted_data[-(chunk_size+k):-chunk_size] if chunk_size > 0 else sorted_data[-k:]
            },
            'fact': {
                'high': sorted_data[2*chunk_size:2*chunk_size+k],
                'low': sorted_data[-(2*chunk_size+k):-(chunk_size)] if chunk_size > 0 else sorted_data[-k:]
            },
            'code': {
                'high': sorted_data[3*chunk_size:3*chunk_size+k],
                'low': sorted_data[-(3*chunk_size+k):-2*chunk_size] if chunk_size > 0 else sorted_data[-k:]
            }
        }
        
        return multidomain
    
    def multidomain_contrastive_reasoning(self,
                                         tasks: Dict[str, str],
                                         multidomain_examples: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> str:
        """Ask LLM: what prompt properties work across ALL domains?"""
        
        # Format examples from all domains
        examples_text = ""
        for domain, examples in multidomain_examples.items():
            high = examples['high'][:2]
            low = examples['low'][:2]
            
            examples_text += f"\n{domain.upper()} DOMAIN:\n"
            examples_text += f"  High quality: {high[0].get('prompt', '')[:60]}...\n"
            examples_text += f"  Low quality: {low[0].get('prompt', '')[:60]}...\n"
        
        reasoning_prompt = f"""
You are a prompt optimization expert specializing in cross-domain effectiveness.

Analyze these high vs low quality examples across MULTIPLE reasoning domains:

{examples_text}

These tasks span different domains:
- Mathematical reasoning: {tasks.get('math', 'unknown')}
- Logical reasoning: {tasks.get('reasoning', 'unknown')}
- Fact verification: {tasks.get('fact', 'unknown')}
- Code generation: {tasks.get('code', 'unknown')}

CRITICAL QUESTION: What prompt properties work well ACROSS ALL these diverse domains?

Consider:
1. What reasoning style is universal (math AND logic AND facts AND code)?
2. What structure helps in all domains?
3. What specific phrases or instructions generalize?
4. What should a robust prompt emphasize?
5. How can we balance specificity with generality?

Please provide 4-5 insights about GENERALIZABLE prompt properties that work across all domains.
"""
        
        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": reasoning_prompt}],
            temperature=TEMPERATURE,
            max_tokens=400
        )
        
        self.api_calls += 1
        reasoning = response.choices[0].message.content
        
        return reasoning
    
    def generate_multidomain_optimized_prompt(self,
                                             reasoning: str) -> str:
        """Generate prompt optimized for robustness across domains."""
        
        generation_prompt = f"""
You are a prompt engineer creating a ROBUST prompt that works across diverse domains.

Based on these insights about universal prompt properties:

{reasoning}

Create ONE optimized prompt that:
1. Incorporates these cross-domain insights
2. Works well for mathematics AND logic AND facts AND code
3. Emphasizes principles that apply everywhere
4. Balances generalization with effectiveness
5. Uses ONLY the placeholder {{question}} - no specific examples
6. Encourages step-by-step thinking and clear reasoning
7. Is clear, specific, and encouraging

The prompt should be generic enough to work for ANY problem in ANY of these domains.

Output ONLY the optimized prompt template, no explanation.
"""
        
        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": generation_prompt}],
            temperature=0.3,
            max_tokens=250
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
    
    def optimize_multidomain(self,
                           helpsteer2_data: List[Dict[str, Any]],
                           tasks: Dict[str, str]) -> Dict[str, Any]:
        """Full multi-domain CRPO pipeline."""
        
        print(f"\n" + "=" * 70)
        print(f"MULTI-DOMAIN CRPO OPTIMIZATION")
        print(f"Optimizing across: {', '.join(tasks.keys())}")
        print("=" * 70)
        
        # Step 1: Retrieve multi-domain examples
        print("\nStep 1: Retrieving multi-domain reference examples...")
        multidomain_examples = self.retrieve_multidomain_examples(helpsteer2_data, k=3)
        print(f"   Retrieved examples from {len(multidomain_examples)} domains")
        
        # Step 2: Multi-domain reasoning
        print("\nStep 2: Contrastive reasoning across domains...")
        reasoning = self.multidomain_contrastive_reasoning(tasks, multidomain_examples)
        print(f"   Generated reasoning (length: {len(reasoning)} chars)")
        print(f"   Sample: {reasoning[:200]}...")
        
        # Step 3: Generate robust prompt
        print("\nStep 3: Generating multi-domain optimized prompt...")
        optimized_prompt = self.generate_multidomain_optimized_prompt(reasoning)
        print(f"   Generated prompt length: {len(optimized_prompt)} chars")
        print(f"   Prompt:\n   {optimized_prompt[:300]}...")
        
        print("\n" + "=" * 70)
        print(f"MULTI-DOMAIN OPTIMIZATION COMPLETE")
        print(f"Total API calls: {self.api_calls}")
        print("=" * 70)
        
        return {
            'optimization_type': 'multi_domain',
            'domains': list(tasks.keys()),
            'tasks': tasks,
            'reasoning': reasoning,
            'optimized_prompt': optimized_prompt,
            'api_calls': self.api_calls
        }
