import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# 1. Setup UI
st.set_page_config(page_title="AI Knowledge Base", layout="centered")
st.title("📚 Custom Dataset Chatbot")
st.info("This bot is powered by a pre-loaded PDF dataset using Gemini 1.5 Flash.")

# 2. Configuration & API Key
# For Streamlit Cloud, set GOOGLE_API_KEY in the dashboard secrets
api_key = st.secrets["GOOGLE_API_KEY"]

# 3. Knowledge Base Logic
@st.cache_resource # Keeps the database in memory so it doesn't reload on every click
def initialize_knowledge_base(data_path="data/"):
    """Loads PDFs from folder, chunks them, and creates a FAISS index."""
    if not os.path.exists(data_path):
        st.error(f"Directory '{data_path}' not found! Please create it and add PDFs.")
        return None
    
    # Load all PDFs in the folder
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Create Vector Store (Free Hugging Face Embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store

def get_chat_response(user_question, vector_store):
    """Retrieves relevant docs and generates answer via Gemini."""
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Helpful Answer:"""
    
    # Setup LLM (Gemini)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    # Search for relevant chunks
    docs = vector_store.similarity_search(user_question, k=3)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# 4. Main Execution
# Initialize once
if "vector_store" not in st.session_state:
    with st.spinner("Indexing your dataset..."):
        st.session_state.vector_store = initialize_knowledge_base()

# Chat Interface
if st.session_state.vector_store:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask something about the data:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            response = get_chat_response(prompt, st.session_state.vector_store)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
          
