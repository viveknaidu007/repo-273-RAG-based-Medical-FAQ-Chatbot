# repo-273-RAG-based-Medical-FAQ-Chatbot
## API Usage Notice

OpenAI does not provide free API credits,  Therefore, this project uses the Gemini API for chatbot functionality. Please use your Gemini API key for all chatbot interactions.

## Dataset
The Q&A dataset is downloaded from Kaggle:


## Setup Instructions
1. Activate the Python environment:
	```cmd
	conda activate <yourenv>
	```
2. Install required packages:
	```cmd
	pip install -r requirements.txt
	```
3. Add your Gemini API key in the chatbot configuration .env.

4. run the app.py:
	```cmd
	streamlit run app.py
	```

## Why Gemini?
Due to the lack of free OpenAI API credits, Gemini API is used for this project. 