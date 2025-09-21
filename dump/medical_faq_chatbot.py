import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load dataset (assume CSV in downloaded folder)
dataset_path = "./gvaldenebro_cancer-q-and-a-dataset/cancer_q_and_a.csv"
df = pd.read_csv(dataset_path)

# Prepare documents for retrieval
texts = df['question'] + "\n" + df['answer']
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents(texts.tolist())

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up RAG QA chain
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Example interaction
query = "What are common symptoms of cancer?"
result = qa_chain.run(query)
print("Chatbot answer:", result)
