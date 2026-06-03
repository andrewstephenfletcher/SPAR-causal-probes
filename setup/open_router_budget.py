import os, requests
from dotenv import load_dotenv
load_dotenv('.env')

key = os.environ['OPENROUTER_API_KEY']
resp = requests.get('https://openrouter.ai/api/v1/auth/key', headers={'Authorization': f'Bearer {key}'})
data = resp.json()['data']
limit = data.get('limit')
usage = data.get('usage')
remaining = (limit - usage) if limit else None
print(f'Limit:     \${limit:.2f}' if limit else 'Limit:     unlimited')
print(f'Usage:     \${usage:.2f}')
print(f'Remaining: \${remaining:.2f}' if remaining is not None else 'Remaining: N/A')