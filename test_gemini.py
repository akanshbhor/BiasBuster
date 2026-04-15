from dotenv import load_dotenv
load_dotenv('backend/.env')
from google import genai
import os
import traceback

try:
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or ''
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model='gemini-2.0-flash', contents='Hello')
    print('TYPE:', type(response))
    print('DIR:', dir(response))
    if hasattr(response, 'text'):
        print('TEXT:', response.text)
    else:
        print('NO TEXT ATTRIBUTE')
except Exception as e:
    print(traceback.format_exc())
