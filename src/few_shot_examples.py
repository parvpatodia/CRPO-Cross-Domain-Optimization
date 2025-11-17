# Few-shot examples for each domain

FEW_SHOT_EXAMPLES = {
    'math': """
Example 1:
Question: If a book costs $12 and you buy 3 books, how much do you spend?
Answer: Let's think step by step. One book costs $12. I buy 3 books. Total = 12 × 3 = $36.

Example 2:
Question: Sarah has 15 apples. She gives 4 to her friend. How many does she have left?
Answer: Let's think step by step. Sarah starts with 15 apples. She gives away 4. Left = 15 - 4 = 11 apples.

Example 3:
Question: A train travels 60 miles per hour for 2 hours. How far does it travel?
Answer: Let's think step by step. Speed = 60 mph, Time = 2 hours. Distance = Speed × Time = 60 × 2 = 120 miles.

Now solve:
Question: {question}
Answer:
""",
    
    'reasoning': """
Example 1:
Question: All dogs are animals. All animals breathe. Are all dogs breathing?
Answer: Let's think step by step. Given: All dogs are animals, and all animals breathe. Therefore, all dogs must be breathing. Answer: True.

Example 2:
Question: Some birds can fly. All flying creatures have wings. Can all birds fly?
Answer: Let's think step by step. Some birds can fly (but not all). We cannot conclude all birds can fly from this information. Answer: False.

Example 3:
Question: If all roses are flowers and some flowers are red, are all roses red?
Answer: Let's think step by step. We know all roses are flowers, but only some flowers are red. We cannot conclude all roses are red. Answer: False.

Now answer:
Question: {question}
Answer:
""",
    
    'fact_verification': """
Example 1:
Statement: Paris is the capital of France.
Answer: Let's think step by step. Paris is widely known and documented as the capital of France. This is confirmed by official records. Answer: True.

Example 2:
Statement: The moon is made of cheese.
Answer: Let's think step by step. The moon is a celestial body made of rock and dust, not cheese. This is a common misconception. Answer: False.

Example 3:
Statement: Water boils at 100 degrees Celsius at sea level.
Answer: Let's think step by step. Water does boil at approximately 100°C at sea level (standard atmospheric pressure). This is scientifically accurate. Answer: True.

Now answer:
Statement: {question}
Answer:
""",
    
    'code': """
Example 1:
Write a function that returns the sum of two numbers.
Answer:
def add(a, b):
    return a + b

Example 2:
Write a function that checks if a number is even.
Answer:
def is_even(n):
    return n % 2 == 0

Example 3:
Write a function that returns the factorial of a number.
Answer:
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

Now write code for:
{question}
Answer:
"""
}
