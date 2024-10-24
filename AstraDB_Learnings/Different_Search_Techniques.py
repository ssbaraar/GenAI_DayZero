# Import necessary libraries
from astrapy import DataAPIClient  # For interacting with AstraDB
import re  # For regular expressions
import json  # For JSON operations
from langchain_astradb import AstraDBVectorStore  # For vector storage in AstraDB
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings
import os  # For environment variables
from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()

def clean_text(text):
    """
    Cleans the input text by removing excessive whitespace and unwanted characters.
    """
    # Remove non-printable characters
    text = ''.join(filter(lambda x: x.isprintable(), text))
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def display_results_json(results, include_metadata=False):
    """
    Formats and displays the search results in JSON format.
    
    Args:
        results (list): List of documents returned from the search.
        include_metadata (bool): Whether to include metadata fields.
    """
    for doc in results:
        display_doc = {
            "Document ID": doc.get('_id'),
            "Text": clean_text(doc.get('text', '')),
            "Vectorize": doc.get('$vectorize')
        }
        if include_metadata:
            similarity = doc.get('$similarity')
            if similarity:
                display_doc["Similarity Score"] = round(similarity, 4)
        print(json.dumps(display_doc, indent=4))
        print("-" * 80)

# Initialize HuggingFaceEmbeddings
model_name = "intfloat/e5-large-v2"
model_kwargs = {'device': 'cpu'}  # Use 'cuda' if GPU is available
hf_embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Load token from environment variable
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")

# Initialize AstraDBVectorStore
vector_store = AstraDBVectorStore(
    embedding=hf_embeddings,
    collection_name="your_collection_name",
    token=ASTRA_TOKEN,
    api_endpoint="your_api_endpoint"
)

def similarity_search(query, k=1):
    """
    Performs a basic similarity search.
    
    Args:
        query (str): The query string.
        k (int): Number of top results to retrieve.
    
    Returns:
        None
    """
    results = vector_store.similarity_search(query=query, k=k)
    display_results_json(results, include_metadata=False)

def similarity_search_with_filter(query, k=1, filter_dict=None):
    """
    Performs a similarity search with metadata filters.
    
    Args:
        query (str): The query string.
        k (int): Number of top results to retrieve.
        filter_dict (dict, optional): Metadata filters.
    
    Returns:
        None
    """
    results = vector_store.similarity_search(query=query, k=k, filter=filter_dict)
    display_results_json(results, include_metadata=True)

def similarity_search_with_score(query, k=1):
    """
    Performs a similarity search and retrieves scores.
    
    Args:
        query (str): The query string.
        k (int): Number of top results to retrieve.
    
    Returns:
        None
    """
    results = vector_store.similarity_search_with_score(query=query, k=k)
    for doc, score in results:
        display_doc = {
            "Document ID": doc.get('_id'),
            "Text": clean_text(doc.get('text', '')),
            "Vectorize": doc.get('$vectorize'),
            "Similarity Score": round(score, 4)
        }
        print(json.dumps(display_doc, indent=4))
        print("-" * 80)

def max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5):
    """
    Performs a maximal marginal relevance search.
    
    Args:
        query (str): The query string.
        k (int): Number of top results to retrieve.
        fetch_k (int): Number of documents to fetch to pass to MMR algorithm.
        lambda_mult (float): Degree of diversity among the results.
    
    Returns:
        None
    """
    results = vector_store.max_marginal_relevance_search(query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    display_results_json(results, include_metadata=True)

def similarity_search_by_vector(embedding, k=1):
    """
    Performs a similarity search by vector.
    
    Args:
        embedding (list): The embedding vector.
        k (int): Number of top results to retrieve.
    
    Returns:
        None
    """
    results = vector_store.similarity_search_by_vector(embedding=embedding, k=k)
    display_results_json(results, include_metadata=True)

def similarity_search_with_score_by_vector(embedding, k=1):
    """
    Performs a similarity search by vector and retrieves scores.
    
    Args:
        embedding (list): The embedding vector.
        k (int): Number of top results to retrieve.
    
    Returns:
        None
    """
    results = vector_store.similarity_search_with_score_by_vector(embedding=embedding, k=k)
    for doc, score in results:
        display_doc = {
            "Document ID": doc.get('_id'),
            "Text": clean_text(doc.get('text', '')),
            "Vectorize": doc.get('$vectorize'),
            "Similarity Score": round(score, 4)
        }
        print(json.dumps(display_doc, indent=4))
        print("-" * 80)

# Example Usage

if __name__ == "__main__":
    query = "example query"
    
    # Basic Similarity Search
    print(f"\nSimilarity search results for '{query}':")
    similarity_search(query=query, k=1)
    
    # Similarity Search with Filter
    print(f"\nSimilarity search results with filter {{'key': 'value'}} for '{query}':")
    similarity_search_with_filter(query=query, k=1, filter_dict={"key": "value"})
    
    # Similarity Search with Score
    print(f"\nSimilarity search results with score for '{query}':")
    similarity_search_with_score(query=query, k=1)
    
    # Maximal Marginal Relevance Search
    print(f"\nMaximal marginal relevance search results for '{query}':")
    max_marginal_relevance_search(query=query, k=4, fetch_k=20, lambda_mult=0.5)
    
    # Similarity Search by Vector
    embedding = hf_embeddings.embed_query(query)
    print(f"\nSimilarity search by vector results for '{query}':")
    similarity_search_by_vector(embedding=embedding, k=1)
    
    # Similarity Search with Score by Vector
    print(f"\nSimilarity search with score by vector results for '{query}':")
    similarity_search_with_score_by_vector(embedding=embedding, k=1)