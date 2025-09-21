# Design Choices and Final Touches for Medical FAQ Chatbot

## RAG Pipeline
- **Retrieval-Augmented Generation (RAG):**
  - The app uses LangChain's RAG pipeline: questions are embedded and indexed using FAISS, and Gemini API generates answers based on retrieved context.
  - The pipeline loads Q&A pairs from `train.csv`, splits them into chunks (now with a large chunk size to avoid warnings), and builds a vector store for semantic search.
  - When a user asks a question, relevant chunks are retrieved and passed to Gemini for answer generation.

## Dataset Handling
- **Custom Dataset:**
  - Useing the `train.csv` in the `datasets` folder, supporting any medical Q&A pairs.
  - Handles both standard and fallback column names for robustness.

## Streamlit UI
- **User Experience:**
  - Simple input box for questions and clear display of answers.
  - The app is ready for further UI improvements (e.g., history, context display, etc.).

## Gemini API
- **API Key Handling:**
  - Loads Gemini API key securely from `.env` using `python-dotenv`.
  - Uses the latest compatible import paths for GoogleGenerativeAIEmbeddings and FAISS.

---

