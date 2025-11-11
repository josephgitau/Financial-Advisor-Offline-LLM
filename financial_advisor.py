# import libraries
import requests
import json

# define ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Create our system prompt
SYSTEM_PROMPT = """You are a helpful financial advisor. Provide clear and concise financial advice based on the user's questions.
Always include: "I am not a licensed financial advisor. Please consult with a professional for personalized advice." at the end of your response.
provide practical, beginner-friendly advice, and avoid complex jargon.
"""

def query_ollama(prompt: str) -> str:
    """
    Query the Ollama API with the provided prompt.
    """
    
    payload = {
        "model": "qwen3:4b",
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 500,
            "num_ctx": 2048
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return data.get("response", "No response from Ollama API.")
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to the Ollama API. Please ensure it is running."
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama API: {str(e)}"
    