import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

PROJECT_ROOT = os.path.dirname(__file__)
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
df = pd.read_csv(DATASET_PATH)
# use 'Question' and 'Answer' columns from train.csv
if 'Question' in df.columns and 'Answer' in df.columns:
    texts = df['Question'].astype(str) + "\n" + df['Answer'].astype(str)
else:
    texts = df.iloc[:,1].astype(str) + "\n" + df.iloc[:,2].astype(str)
text_splitter = CharacterTextSplitter(chunk_size=25000, chunk_overlap=50)
docs = text_splitter.create_documents(texts.tolist())
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

st.title("Medical FAQ Chatbot")
with st.form("qa_form"):
    user_query = st.text_input("Ask a medical questions")
    submitted = st.form_submit_button("Submit")
    if submitted and user_query:
        answer = qa_chain.run(user_query)
        st.markdown("**Your Question:**")
        st.write(user_query)
        st.markdown("**Answer:**")
        st.write(answer)
