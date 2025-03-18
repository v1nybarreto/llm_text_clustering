import os
import re
import numpy as np
import zipfile
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import spacy
import fitz  # PyMuPDF
import plotly.express as px
import warnings
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from openai import OpenAI
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Load language model for English lemmatization
nlp = spacy.load("en_core_web_sm")

# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Load API key from .env file

# 1. Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text or None if an error occurs.
    """
    try:
        doc = fitz.open(pdf_path)  # Use fitz.open to open the PDF
        text = ""
        for page in doc:
            text += page.get_text()
        print(f"Text extraction successful for {pdf_path}")
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

# 2. Function to preprocess text
def preprocess_text(text):
    """
    Preprocesses the text by removing punctuation, numbers, stopwords, and performing lemmatization.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Preprocessed text or None if an error occurs.
    """
    try:
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        doc = nlp(text)  # Lemmatization and stopword removal
        text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        print("Text preprocessing completed.")
        return text
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# 3. Function to generate embeddings
def generate_embeddings(texts, model_name="all-mpnet-base-v2"):
    """
    Generates embeddings for a list of texts using a SentenceTransformer model.
    
    Args:
        texts (list): List of texts to generate embeddings for.
        model_name (str): Name of the SentenceTransformer model.
    
    Returns:
        np.ndarray: Embeddings or None if an error occurs.
    """
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts)
        print("Embeddings generated successfully.")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

# 4. Function to reduce dimensionality
def reduce_dimensionality(embeddings, method="umap"):
    """
    Reduces the dimensionality of embeddings using UMAP, t-SNE, or PCA.
    
    Args:
        embeddings (np.ndarray): High-dimensional embeddings.
        method (str): Dimensionality reduction method ("umap", "tsne", or "pca").
    
    Returns:
        np.ndarray: 2D embeddings or None if an error occurs.
    """
    try:
        if method == "umap":
            reducer = UMAP(random_state=42, n_components=2, n_jobs=1)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        print(f"Dimensionality reduction with {method.upper()} completed.")
        return embeddings_2d
    except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        return None

# 5. Function to cluster texts
def cluster_texts(embeddings, method="hdbscan"):
    """
    Clusters texts using HDBSCAN, DBSCAN, or KMeans.
    
    Args:
        embeddings (np.ndarray): Embeddings to cluster.
        method (str): Clustering method ("hdbscan", "dbscan", or "kmeans").
    
    Returns:
        np.ndarray: Cluster labels or None if an error occurs.
    """
    try:
        if method == "hdbscan":
            clusterer = HDBSCAN(min_cluster_size=3, min_samples=2, cluster_selection_method='leaf')
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        elif method == "kmeans":
            clusterer = KMeans(n_clusters=5, random_state=42)
        clusters = clusterer.fit_predict(embeddings)
        print(f"Clustering with {method.upper()} completed.")
        return clusters
    except Exception as e:
        print(f"Error during clustering: {e}")
        return None

# 6. Function to calculate Silhouette Score
def calculate_silhouette_score(embeddings, clusters):
    """
    Calculates the Silhouette Score for clustering results, excluding outliers.
    
    Args:
        embeddings (np.ndarray): Embeddings used for clustering.
        clusters (np.ndarray): Cluster labels.
    
    Returns:
        float: Silhouette Score or None if not enough clusters are present.
    """
    try:
        # Filter embeddings and clusters to remove outliers (label -1)
        valid_indices = clusters != -1
        valid_embeddings = embeddings[valid_indices]
        valid_clusters = clusters[valid_indices]

        # Calculate Silhouette Score only if there are at least 2 valid clusters
        if len(set(valid_clusters)) > 1:
            score = silhouette_score(valid_embeddings, valid_clusters)
            return score
        else:
            print("Not enough clusters to calculate Silhouette Score.")
            return None
    except Exception as e:
        print(f"Error calculating Silhouette Score: {e}")
        return None

# 7. Function to filter and group clusters by similarity
def filter_and_group_clusters(texts, embeddings, clusters, similarity_threshold=0.7):
    """
    Filters out outliers and groups texts within clusters based on similarity.
    
    Args:
        texts (list): List of texts.
        embeddings (np.ndarray): Embeddings of the texts.
        clusters (np.ndarray): Cluster labels.
        similarity_threshold (float): Threshold for cosine similarity.
    
    Returns:
        dict: Grouped texts by cluster ID or None if an error occurs.
    """
    try:
        # Remove outliers (label -1)
        valid_indices = clusters != -1
        valid_texts = [texts[i] for i in range(len(texts)) if valid_indices[i]]
        valid_embeddings = embeddings[valid_indices]
        valid_clusters = clusters[valid_indices]

        # Group texts by cluster
        grouped_texts = {}
        for cluster_id in set(valid_clusters):
            cluster_indices = valid_clusters == cluster_id
            cluster_texts = [valid_texts[i] for i in range(len(valid_texts)) if cluster_indices[i]]
            cluster_embeddings = valid_embeddings[cluster_indices]

            # Calculate similarity within the cluster
            similarity_matrix = cosine_similarity(cluster_embeddings)
            high_similarity_indices = np.where(similarity_matrix > similarity_threshold)
            grouped_texts[cluster_id] = {
                "texts": cluster_texts,
                "similarity_matrix": similarity_matrix
            }
        print("Outliers removed and texts grouped by similarity.")
        return grouped_texts
    except Exception as e:
        print(f"Error during filtering and grouping: {e}")
        return None

# 8. Function to split large texts
def split_text(text, max_tokens=4000):
    """
    Splits a large text into smaller chunks, ensuring each chunk has at most `max_tokens`.
    
    Args:
        text (str): Input text.
        max_tokens (int): Maximum number of tokens per chunk.
    
    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# 9. Function to summarize large texts
def summarize_large_text(text, model="gpt-4", max_tokens=4000):
    """
    Summarizes large texts by splitting them into smaller chunks and summarizing each chunk.
    
    Args:
        text (str): Input text.
        model (str): OpenAI model to use for summarization.
        max_tokens (int): Maximum number of tokens per chunk.
    
    Returns:
        str: Concatenated summary of all chunks.
    """
    chunks = split_text(text, max_tokens=max_tokens)
    summaries = []

    for chunk in chunks:
        summary = summarize_text(chunk, max_tokens=500)
        if summary:
            summaries.append(summary)

    return " ".join(summaries)

# 10. Function to summarize text using OpenAI
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def summarize_text(text, max_tokens=500):
    """
    Summarizes text using OpenAI's GPT-4 model.
    
    Args:
        text (str): Input text.
        max_tokens (int): Maximum number of tokens for the summary.
    
    Returns:
        str: Summary of the text.
    """
    try:
        # Use GPT-4 to summarize the text
        prompt = (
            "Summarize the following text clearly and concisely, retaining the main points. "
            f"Text:\n{text}\n\n"
            "Summary:"
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a text summarization assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text[:max_tokens]  # Return the beginning of the text if summarization fails

# 11. Function to generate a description for a single PDF
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_pdf_description(text, model="gpt-4", max_tokens=500):
    """
    Generates a thematic description for a single PDF.
    
    Args:
        text (str): Text extracted from the PDF.
        model (str): OpenAI model to use for description generation.
        max_tokens (int): Maximum number of tokens for the description.
    
    Returns:
        str: Thematic description of the PDF.
    """
    try:
        # Summarize the text before sending it to the API
        summarized_text = summarize_large_text(text, model=model)

        prompt = (
            "You are a text analysis assistant. "
            "Based on the provided text, generate a general thematic description "
            "summarizing the main topic and relevant subtopics. "
            "Be clear and concise.\n\n"
            f"Text:\n{summarized_text}\n\n"
            "Thematic Description:"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a text analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Adjust as needed
            temperature=0.7,
        )

        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        print(f"Error generating PDF description: {e}")
        raise

# 12. Function to generate a cluster description from individual PDF descriptions
def generate_cluster_description_from_pdfs(texts, model="gpt-4", rate_limit_tokens=90000, tokens_per_minute=90000):
    """
    Generates a thematic description for a cluster based on individual PDF descriptions.
    Controls the request rate to avoid exceeding the token limit per minute.
    
    Args:
        texts (list): List of texts from PDFs in the cluster.
        model (str): OpenAI model to use for description generation.
        rate_limit_tokens (int): Token limit per minute.
        tokens_per_minute (int): Tokens allowed per minute.
    
    Returns:
        str: Thematic description of the cluster.
    """
    try:
        # Generate individual descriptions for each PDF
        pdf_descriptions = []
        total_tokens = 0

        for text in texts:
            # Check if the token limit per minute is reached
            if total_tokens >= rate_limit_tokens:
                print("Token limit per minute reached. Waiting for 60 seconds...")
                time.sleep(60)  # Wait for 1 minute
                total_tokens = 0  # Reset token counter

            # Generate PDF description
            description = generate_pdf_description(text, model=model)
            if description:
                pdf_descriptions.append(description)
                total_tokens += len(description.split())  # Estimate tokens (1 token ~= 1 word)

        if not pdf_descriptions:
            print("No PDF descriptions were generated.")
            return None

        # Concatenate individual descriptions
        combined_descriptions = " ".join(pdf_descriptions)

        # Summarize the concatenated descriptions into a final concise description
        prompt = (
            "You are a text analysis assistant. "
            "Based on the following thematic descriptions of multiple PDFs, generate a general description "
            "summarizing the main topics and relevant subtopics. "
            "Be clear and concise.\n\n"
            f"Descriptions:\n{combined_descriptions}\n\n"
            "General Thematic Description:"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a text analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # Adjust as needed
            temperature=0.7,
        )

        final_description = response.choices[0].message.content.strip()
        return final_description
    except Exception as e:
        print(f"Error generating cluster description: {e}")
        return None

# 13. Function to generate descriptions for all clusters
def generate_all_cluster_descriptions(grouped_texts, model="gpt-4"):
    """
    Generates thematic descriptions for all clusters.
    
    Args:
        grouped_texts (dict): Texts grouped by cluster ID.
        model (str): OpenAI model to use for description generation.
    
    Returns:
        dict: Thematic descriptions for each cluster.
    """
    cluster_descriptions = {}
    for cluster_id, data in grouped_texts.items():
        print(f"Generating description for Cluster {cluster_id}...")
        description = generate_cluster_description_from_pdfs(data["texts"], model=model)
        if description:
            cluster_descriptions[cluster_id] = description
            print(f"Cluster {cluster_id} Description: {description}\n")
        else:
            print(f"Failed to generate description for Cluster {cluster_id}.\n")
    return cluster_descriptions

# 14. Function to visualize clusters interactively
def visualize_clusters_interactive(grouped_texts):
    """
    Visualizes clusters interactively using Plotly.
    
    Args:
        grouped_texts (dict): Texts grouped by cluster ID.
    """
    try:
        # Extract cluster information
        cluster_ids = []
        avg_similarities = []
        for cluster_id, data in grouped_texts.items():
            cluster_ids.append(cluster_id)
            avg_similarities.append(np.mean(data["similarity_matrix"]))

        # Create DataFrame for visualization
        df = pd.DataFrame({
            "Cluster": cluster_ids,
            "Average Similarity": avg_similarities
        })

        # Create scatter plot with circles
        fig = px.scatter(df, x="Cluster", y="Average Similarity",
                         title="Cluster Visualization",
                         labels={"Cluster": "Cluster Number", "Average Similarity": "Average Similarity"},
                         text="Average Similarity")
        fig.update_traces(marker=dict(size=12, symbol="circle"),  # Set symbol to circle
                          textposition="top center")  # Position text above circles
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Ensure all clusters are shown on the X-axis
        fig.show()
        print("Cluster visualization with circles completed.")
    except Exception as e:
        print(f"Error during cluster visualization: {e}")

# 15. Function to show cluster information
def show_cluster_info(grouped_texts, clusters):
    """
    Displays information about clusters, including the number of outliers.
    
    Args:
        grouped_texts (dict): Texts grouped by cluster ID.
        clusters (np.ndarray): Cluster labels.
    """
    try:
        # Count outliers
        num_outliers = np.sum(clusters == -1)
        print(f"\nNumber of outliers: {num_outliers}")

        # Display information for each cluster
        print("\nCluster Information:")
        for cluster_id, data in grouped_texts.items():
            num_files = len(data["texts"])
            avg_similarity = np.mean(data["similarity_matrix"])
            print(f"Cluster {cluster_id}: {num_files} files, Average Similarity: {avg_similarity:.4f}")
    except Exception as e:
        print(f"Error showing cluster info: {e}")

# 16. Main pipeline function
def pipeline(pdf_folder, similarity_threshold=0.7, model="gpt-4"):
    """
    Executes the entire pipeline: text extraction, preprocessing, clustering, and visualization.
    
    Args:
        pdf_folder (str): Path to the folder containing PDF files.
        similarity_threshold (float): Threshold for cosine similarity.
        model (str): OpenAI model to use for summarization and description generation.
    """
    # List all PDF files in the folder
    pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_paths:
        print(f"No PDF files found in {pdf_folder}.")
        return

    texts = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        if text:
            preprocessed_text = preprocess_text(text)
            if preprocessed_text:
                texts.append(preprocessed_text)

    if not texts:
        print("No valid texts found for processing.")
        return

    # Generate embeddings
    embeddings = generate_embeddings(texts)
    if embeddings is None:
        return

    # Reduce dimensionality with UMAP
    embeddings_2d = reduce_dimensionality(embeddings, method="umap")
    if embeddings_2d is None:
        return

    # Cluster texts
    clusters = cluster_texts(embeddings, method="hdbscan")
    if clusters is None:
        return

    # Calculate Silhouette Score
    _ = calculate_silhouette_score(embeddings, clusters)

    # Remove outliers and group texts by similarity
    grouped_texts = filter_and_group_clusters(texts, embeddings, clusters, similarity_threshold)
    if grouped_texts is None:
        return

    # Generate thematic descriptions for clusters
    cluster_descriptions = generate_all_cluster_descriptions(grouped_texts, model=model)

    # Show cluster information
    show_cluster_info(grouped_texts, clusters)

    # Visualize clusters interactively with circles
    visualize_clusters_interactive(grouped_texts)

    # Display thematic descriptions of clusters
    print("\nThematic Descriptions of Clusters:")
    for cluster_id, description in cluster_descriptions.items():
        print(f"Cluster {cluster_id}: {description}")

# Example usage
if __name__ == "__main__":
    pdf_folder = "path/to/pdf/folder"  # Replace with the path to your PDF folder
    pipeline(pdf_folder, similarity_threshold=0.7, model="gpt-4")