import google.generativeai as genai
import os

# Set your Gemini API key here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
genai.configure(api_key=GEMINI_API_KEY)

# Example: Generate a response
model = genai.GenerativeModel('gemini-pro')
prompt = "What are common questions about cancer?"
response = model.generate_content(prompt)
print("Gemini Response:", response.text)
