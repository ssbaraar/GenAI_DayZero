# GenAI_DayZero: Vector Search Strategies using AstraDB and LangChain

This repository demonstrates a hands-on implementation of various **vector search strategies** using **AstraDB** for vector storage and **LangChain** integrated with **HuggingFace Embeddings**. It includes examples of similarity search, filtered search, similarity with score, and maximal marginal relevance search, along with their respective use cases.

## Table of Contents
- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Environment Setup](#environment-setup)
- [Code Overview](#code-overview)
  - [Cleaning Text](#cleaning-text)
  - [Displaying Results in JSON](#displaying-results-in-json)
  - [Vector Search Strategies](#vector-search-strategies)
    - [Basic Similarity Search](#basic-similarity-search)
    - [Similarity Search with Filter](#similarity-search-with-filter)
    - [Similarity Search with Scores](#similarity-search-with-scores)
    - [Maximal Marginal Relevance (MMR) Search](#maximal-marginal-relevance-mmr-search)
    - [Vector Search by Embedding](#vector-search-by-embedding)
- [Usage](#usage)
- [References](#references)

## Introduction
This project is part of the **GenAI_DayZero** initiative, where we explore different techniques related to **Generative AI** and **Large Language Models (LLMs)**. In this project, we focus on vector-based similarity search techniques and their applications in different scenarios such as document retrieval, recommendation systems, and diversity-based searches.

### Key Features:
- Basic vector-based similarity search.
- Filtered similarity search using metadata.
- Similarity search with confidence scores.
- Maximal Marginal Relevance (MMR) search for diversified results.
- Pre-computed embedding-based search.

## Tech Stack

- **AstraDB**: A highly scalable, serverless NoSQL database that supports vector search with low-latency queries.
  - [AstraDB Documentation](https://docs.datastax.com/en/astra/docs/)

- **LangChain**: A framework for developing applications powered by language models, especially useful for text splitting and vector operations.
  - [LangChain Documentation](https://langchain.readthedocs.io/)

- **HuggingFace Embeddings**: Provides access to state-of-the-art pre-trained models for generating text embeddings.
  - [HuggingFace Documentation](https://huggingface.co/docs)

- **Python**: The primary language used for implementation.
  - [Python Official Docs](https://docs.python.org/)

- **dotenv**: Used for managing environment variables.
  - [dotenv Documentation](https://pypi.org/project/python-dotenv/)

## Environment Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- `pip` for installing dependencies

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GenAI_DayZero.git
   cd GenAI_DayZero
   ```

2. Install the required dependencies:
   ```bash
   pip install astrapy langchain langchain-huggingface python-dotenv
   ```

3. Create a `.env` file in the root of the project and add the following environment variables:
   ```bash
   ASTRA_TOKEN=<Your AstraDB Token>
   ASTRA_COLLECTION_NAME=<Your AstraDB Collection Name>
   ASTRA_API_ENDPOINT=<Your AstraDB API Endpoint>
   ```

4. Load your environment variables by running the following command:
   ```bash
   source .env
   ```

## Code Overview

### Cleaning Text
The function `clean_text(text)` is responsible for cleaning and formatting the input text by removing non-printable characters, extra spaces, and newlines.

```python
def clean_text(text):
    # Cleans input text by removing excessive whitespace and unwanted characters.
    ...
```

### Displaying Results in JSON
The `display_results_json(results, include_metadata=False)` function formats and prints the search results in a structured JSON format. If metadata like similarity scores is available, it is also displayed.

```python
def display_results_json(results, include_metadata=False):
    # Displays search results in a clean, readable JSON format.
    ...
```

### Vector Search Strategies

#### 1. Basic Similarity Search
This method compares vectors to find the closest matches.

```python
def similarity_search(query, k=1):
    # Performs a basic similarity search using AstraDB.
    ...
```

#### 2. Similarity Search with Filter
This method combines vector similarity with metadata filtering, making it useful for constrained searches, like searching within specific categories or timeframes.

```python
def similarity_search_with_filter(query, k=1, filter_dict=None):
    # Performs similarity search with additional metadata filters.
    ...
```

#### 3. Similarity Search with Scores
This function returns similarity scores alongside the search results, which can be used in recommendation systems where thresholds matter.

```python
def similarity_search_with_score(query, k=1):
    # Retrieves similarity search results along with confidence scores.
    ...
```

#### 4. Maximal Marginal Relevance (MMR) Search
MMR balances the relevance and diversity of the results, making it useful in cases like research paper recommendations.

```python
def max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5):
    # Balances relevance with diversity to avoid redundant results.
    ...
```

#### 5. Vector Search by Embedding
This search method is useful for scenarios where you already have pre-computed embeddings.

```python
def similarity_search_by_vector(embedding, k=1):
    # Performs a search using pre-computed embeddings.
    ...
```

### Example Usage
Here’s how you can use the various search strategies:

1. **Basic Similarity Search**:
   ```python
   query = "example query"
   similarity_search(query=query, k=1)
   ```

2. **Similarity Search with Filter**:
   ```python
   filter_dict = {"key": "value"}
   similarity_search_with_filter(query=query, k=1, filter_dict=filter_dict)
   ```

3. **Similarity Search with Scores**:
   ```python
   similarity_search_with_score(query=query, k=1)
   ```

4. **Maximal Marginal Relevance (MMR) Search**:
   ```python
   max_marginal_relevance_search(query=query, k=4, fetch_k=20, lambda_mult=0.5)
   ```

5. **Vector Search by Embedding**:
   ```python
   embedding = hf_embeddings.embed_query(query)
   similarity_search_by_vector(embedding=embedding, k=1)
   ```

## References
- [AstraDB Documentation](https://docs.datastax.com/en/home/index.html)
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [Python dotenv Documentation](https://pypi.org/project/python-dotenv/)
