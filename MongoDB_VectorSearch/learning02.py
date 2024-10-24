# pdfprocess.py

import os
import logging
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import google.generativeai as genai
import key_param
import traceback
from pymongo.operations import SearchIndexModel

# Initialize logging
logging.basicConfig(filename="event_log.txt", level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

# Set up Google API for embeddings and Gemini
genai.configure(api_key=key_param.google_api_key)

# Custom Embedding class (unchanged)
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
try:
    client = MongoClient(key_param.MONGO_URI)
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

# Function to extract text from PDFs (unchanged)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks (unchanged)
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to process PDF and store embedded chunks into MongoDB as clusters
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
    
    # Create vector search index
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 768,  # Adjust this if your embedding dimension is different
                    "path": "chunks_and_embeddings.embedding",
                    "similarity": "cosine"
                }
            ]
        },
        name="vector_index",
        type="vectorSearch"
    )
    collection.create_search_index(model=search_index_model)
    print("Vector search index created")

# Function to retrieve similar documents
def retrieve_similar_documents(query, top_k=3):
    custom_embeddings = CustomEmbedding()
    query_embedding = custom_embeddings.embed_query(query)
    
    pipeline = [
        {
            "$search": {
                "index": "vector_index",
                "knnBeta": {
                    "vector": query_embedding,
                    "path": "chunks_and_embeddings.embedding",
                    "k": top_k
                }
            }
        },
        {
            "$project": {
                "text_chunk": "$chunks_and_embeddings.text_chunk",
                "score": {"$meta": "searchScore"}
            }
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    return results

# Function to generate response using Gemini
def generate_response(query, context):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Answer the following question based on the given context. If the answer is not in the context, say "I don't have enough information to answer that question."

    Question: {query}

    Context: {context}

    Answer:
    """
    
    response = model.generate_content(prompt)
    return response.text

# Main function to process query
def process_query(query):
    similar_docs = retrieve_similar_documents(query)
    context = "\n".join([doc['text_chunk'][0] for doc in similar_docs])
    response = generate_response(query, context)
    return response

# Example usage
if __name__ == "__main__":
    pdf_docs = ["ERP Complete Digital notes.pdf"]  # Replace with your actual PDF file paths
    process_and_store_pdf(pdf_docs)
    
    # Example query
    user_query = "What are the main components of an ERP system?"
    answer = process_query(user_query)
    
    print(f"User Query: {user_query}")
    print(f"Generated Response: {answer}")