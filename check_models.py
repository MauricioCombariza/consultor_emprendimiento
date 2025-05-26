import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("API Key de Google no encontrada. Asegúrate de configurar el archivo .env correctamente.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Modelos disponibles que soportan 'generateContent':")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)