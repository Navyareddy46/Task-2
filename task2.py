from bs4 import BeautifulSoup
import requests
import numpy as np
import faiss  # Make sure you import FAISS
from sentence_transformers import SentenceTransformer

# Step 1: Scrape Website Content (using the UND URL)
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract paragraphs and other relevant sections (we're still scraping all 'p' tags here)
    paragraphs = soup.find_all('p')
    content = " ".join([para.get_text() for para in paragraphs])

    return content

# Step 2: Chunk the Content (break into smaller pieces)
def chunk_content(content, chunk_size=1000):
    # Chunk the content into smaller sections
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    return chunks

# Step 3: Embed the Chunks using Sentence Transformer
def embed_chunks(chunks, model):
    embeddings = model.encode(chunks)
    return embeddings

# Step 4: Initialize FAISS Index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Should match the embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
    faiss_index = faiss.IndexFlatL2(dimension)  # FAISS index for cosine similarity search
    faiss_index.add(embeddings)
    return faiss_index

# Step 5: Query Index Function (to search for relevant chunks)
def query_index(query, faiss_index, model, metadata, top_k=5):
    # Embed the query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Perform similarity search using FAISS
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Retrieve the top-k results based on indices
    results = [metadata[idx] for idx in indices[0]]
    return results

# Example usage:

# Step 6: Scrape, chunk, embed, and create FAISS index
url = "https://und.edu/"  # University of North Dakota URL
scraped_content = scrape_website(url)

# Chunk the scraped content
chunks = chunk_content(scraped_content)

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the chunks
chunk_embeddings = embed_chunks(chunks, model)

# Create FAISS index from the embeddings
faiss_index = create_faiss_index(chunk_embeddings)

# Step 7: Save metadata (for referencing the chunks)
metadata = [{"text": chunk, "source": url} for chunk in chunks]

# Step 8: Query example: "What programs are offered at the University of North Dakota?"
query = "What programs are offered at the University of North Dakota?"
retrieved_results = query_index(query, faiss_index, model, metadata)

# Print the top-k results
for result in retrieved_results:
    print(f"Source: {result['source']}\nText: {result['text']}\n")