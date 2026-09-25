import argparse
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()
    api_key = os.getenv('OPENROUTER_API_KEY')
    model = os.getenv('OPENROUTER_MODEL')
    if not api_key:
        raise RuntimeError('OPENROUTER_API_KEY is missing from .env')
    if not model:
        raise RuntimeError('OPENROUTER_MODEL is missing from .env')
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        json={'model': model, 'messages': [
            {'role': 'system', 'content': 'You are a beginner-friendly Java and DSA tutor. Be accurate and say when you are uncertain.'},
            {'role': 'user', 'content': args.prompt}
        ]},
        timeout=60,
    )
    response.raise_for_status()
    print(response.json()['choices'][0]['message']['content'])

if __name__ == '__main__':
    main()