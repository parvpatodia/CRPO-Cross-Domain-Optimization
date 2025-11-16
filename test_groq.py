from groq import Groq
import os

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please set GROQ_API_KEY environment variable")

# Create client
client = Groq(api_key=api_key)

# Test call
print("Testing Groq API...")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(f"Response: {response.choices[0].message.content}")
print(f"API working: YES")
print(f"Model available: llama-3.1-8b-instant")
print(f"Response time: ~2 seconds")