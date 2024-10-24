# pdfprocess.py

import os
import logging
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import google.generativeai as genai
import key_param  # Assuming this is a separate file for storing sensitive information
import traceback

# Initialize logging
logging.basicConfig(filename="event_log.txt", level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

# Set up Google API for embeddings
# Note: Ensure key_param.google_api_key is securely stored and not exposed
genai.configure(api_key=key_param.google_api_key)

# Custom Embedding class
# Learning: Custom embedding allows flexibility in choosing embedding models
class CustomEmbedding():
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text):
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Embedding of text"
        )
        return result['embedding']

# Set up MongoDB connection
# Learning: Always use try-except for database connections to handle potential errors
try:
    client = MongoClient(key_param.MONGO_URI)  # Ensure MONGO_URI is securely stored
    dbName = "pdf_demo"
    collectionName = "embedded_pdf_texts"
    collection = client[dbName][collectionName]
    print("Successfully connected to MongoDB")
    # Clear the collection for clean insertion (optional, remove if not needed)
    collection.delete_many({})
    print("Cleared existing documents from the collection")
except Exception as e:
    print(f"Error connecting to MongoDB or clearing collection: {e}")
    exit(1)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
# Learning: Chunking is crucial for efficient processing and embedding of large texts
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to process PDF and store embedded chunks into MongoDB as clusters
# Learning: Structuring data with document_id allows for easier retrieval and management
def process_and_store_pdf(pdf_docs):
    load_dotenv()
    custom_embeddings = CustomEmbedding()
    for pdf_path in pdf_docs:
        raw_text = get_pdf_text([pdf_path])
        text_chunks = get_text_chunks(raw_text)
        logging.info(f"Text extracted and split into chunks for {pdf_path}")
        print(f"Text extracted and split into chunks for {pdf_path}")
        document_id = str(uuid.uuid4())
        chunk_embeddings = []
        for chunk in text_chunks:
            embedding = custom_embeddings.embed_query(chunk)
            chunk_embeddings.append({
                "text_chunk": chunk,
                "embedding": embedding
            })
        document_data = {
            "document_id": document_id,
            "file_name": os.path.basename(pdf_path),
            "chunks_and_embeddings": chunk_embeddings
        }
        try:
            collection.insert_one(document_data)
            print(f"Successfully stored document {document_id} in MongoDB")
            logging.info(f"Successfully stored document {document_id} in MongoDB")
        except Exception as e:
            print(f"Error storing document {document_id} in MongoDB: {e}")
            logging.error(f"Error storing document {document_id} in MongoDB: {traceback.format_exc()}")

# Example usage
if __name__ == "__main__":
    pdf_docs = ["sample.pdf"]  # Replace with your actual PDF file paths
    process_and_store_pdf(pdf_docs)

# Learning: This script demonstrates a complete pipeline for:
# 1. Extracting text from PDFs
# 2. Splitting text into manageable chunks
# 3. Generating embeddings for each chunk
# 4. Storing the structured data in MongoDB
# This approach enables efficient vector search and retrieval of relevant PDF content.