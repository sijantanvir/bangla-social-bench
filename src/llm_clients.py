import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

def call_gpt(user_input, model="gpt-4o"):
    """
    Sends a prompt to GPT and returns the raw text response and logprobs.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
        
    client = OpenAI(api_key=api_key)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": user_input}],
            # Uncomment below if you need logprobs for this specific script
            # logprobs=True,
            # top_logprobs=10
        )
        msg = completion.choices[0]
        text = msg.message.content.strip()
        
        # logprobs = msg.logprobs.model_dump() if msg.logprobs else None
        # return text, logprobs
        
        return text

    except Exception as e:
        print(f"GPT error: {e}")
        return ""


def call_gemini(user_input, model="gemini-2.5-flash"):
    """
    Sends a prompt to Gemini via REST API and returns the raw text response.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_api_key}"

    payload = {
        "contents": [{"parts": [{"text": user_input}]}],
        "generationConfig": {
            "temperature": 0.0,
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        content = response.json()

        if "candidates" in content and content["candidates"]:
            return content["candidates"][0]["content"]["parts"][0]["text"].strip()

        print("Gemini: No candidates found.")
        return ""

    except Exception as e:
        print(f"Gemini error: {e}")
        return ""


def call_openrouter(user_input, model):
    """
    Sends a prompt to an OpenRouter-hosted model.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    try:
        completion = openrouter_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_input}]}],
            extra_headers={
                "X-Title": "BanglaSocialBench",
            }
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenRouter error: {e}")
        return ""


def call_together(user_input, model="meta-llama/Meta-Llama-3-8B-Instruct-Lite"):
    """
    Sends a prompt to TogetherAI.
    """
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY is not set.")

    together_client = OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )
    
    try:
        response = together_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_input}],
            max_tokens=50,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        choice = response.choices[0]
        text = choice.message.content.strip()
        return text, choice.logprobs
        
    except Exception as e:
        print(f"TogetherAI error: {e}")
        return "", None