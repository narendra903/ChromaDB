import streamlit as st
from dotenv import load_dotenv
import os
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.chroma import ChromaDB
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3



# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


# Configuration for ChromaDB
PERSIST_DIRECTORY = "./chromadb_store"
COLLECTION_NAME = "user_pdf"

# Initialize ChromaDB with Gemini for embeddings.
vector_db = ChromaDB(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedder=GeminiEmbedder()  # Uses the API key from your .env file
)

st.title("Upload PDF and Ask Questions")
st.write("Upload a PDF file, and the agent will answer your questions based on its content.")

# File uploader for PDFs
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file locally
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded file saved as {file_path}")

    # Create the knowledge base using the local file's URL.
    # Depending on the implementation, you might need to adjust the knowledge base to support file paths.
    knowledge_base = PDFUrlKnowledgeBase(
        urls=[f"http://localhost:8501/uploads/{uploaded_file.name}"],  # If you serve the file via HTTP
        vector_db=vector_db,
    )

    # Alternatively, if Agno supports processing local file paths directly, you might do:
    # knowledge_base.load_from_file(file_path)

    if st.button("Process PDF"):
        with st.spinner("Processing PDF and generating embeddings..."):
            knowledge_base.load(recreate=True)
        st.success("PDF processed and knowledge base updated!")

    # Create the agent using the Gemini model.
    agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        knowledge=knowledge_base,
        show_tool_calls=True,
    )

    query = st.text_input("Ask a question about the uploaded PDF:")
    if st.button("Get Response") and query:
        with st.spinner("Generating response..."):
            response = agent.print_response(query, markdown=True)
        st.markdown(response)