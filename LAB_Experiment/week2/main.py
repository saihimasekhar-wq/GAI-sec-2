
import os
from dotenv import load_dotenv

from llm_client import LLMClient, GeminiProvider, OllamaProvider

load_dotenv()

print("Choose your LLM Provider:")
print("1. Gemini")
print("2. Ollama (Llama 3.2)")

choice = input("Enter your choice (1 or 2): ")

if choice == "1":
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    provider = GeminiProvider(api_key)

elif choice == "2":
    provider = OllamaProvider()

else:
    raise ValueError("Invalid choice. Please enter 1 or 2.")

client = LLMClient(provider)

prompt = input("\nEnter your prompt: ")

response = client.generate(prompt)

print("\nLLM Response:")
print(response)
