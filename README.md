# LLM Text Clustering

This project aims to cluster text documents (PDFs) into thematic groups using advanced Natural Language Processing (NLP) techniques and Large Language Models (LLMs). The project is divided into several stages, from text extraction and preprocessing to clustering, visualization, and thematic description generation. All steps are implemented within a single Python script: `llm_text_clustering.py`.

## Project Structure

```
llm_text_clustering/
│
├── source/                         # Folder containing the main script
│   └── llm_text_clustering.py      # Main script with all steps
├── data/                           # Folder containing PDF files for analysis
│   └── llm_text_clustering_data/
├── results/                        # Folder to store results and visualizations
│   └── Figure_I/                   # Folder containing cluster visualizations
├── README.md                       # This file
└── requirements.txt                # List of dependencies
```

## Requirements

To run this project, you need the following Python packages:

- pandas
- numpy
- matplotlib
- scikit-learn
- sentence-transformers
- hdbscan
- umap-learn
- spacy
- pymupdf
- plotly
- openai
- python-dotenv

Install the packages using:

```bash
pip install -r requirements.txt
```

Additionally, you will need an OpenAI API key to use GPT-4 for text summarization and thematic description generation. Add your API key to a `.env` file in the root directory:

```
OPENAI_API_KEY=your_api_key_here
```

## Steps

The project follows these steps:

1. **Text Extraction from PDFs:**  
   - Text is extracted from PDF files using the PyMuPDF (fitz) library.
   - The extracted text is preprocessed to remove punctuation, numbers, stopwords, and to perform lemmatization.

2. **Embedding Generation:**  
   - Preprocessed texts are converted into embeddings using the SentenceTransformer model (e.g., `all-mpnet-base-v2`).
   - These embeddings capture the semantic meaning of the texts.

3. **Dimensionality Reduction:**  
   - The dimensionality of the embeddings is reduced using UMAP, t-SNE, or PCA for easier visualization and clustering.

4. **Text Clustering:**  
   - Texts are clustered using algorithms such as HDBSCAN, DBSCAN, or KMeans.
   - HDBSCAN is preferred for its ability to handle outliers and automatically identify clusters.

5. **Silhouette Score Calculation:**  
   - The quality of the clustering is evaluated using the Silhouette Score, which measures how similar texts are within clusters compared to other clusters.

6. **Outlier Removal and Similarity-Based Grouping:**  
   - Outliers (texts not assigned to any cluster) are removed.
   - Texts within each cluster are grouped based on cosine similarity.

7. **Thematic Description Generation:**  
   - Using OpenAI's GPT-4, thematic descriptions are generated for each cluster based on the individual PDF descriptions.

8. **Interactive Cluster Visualization:**  
   - Clusters are visualized interactively using Plotly, with circles representing clusters and their average similarity scores.

9. **Cluster Information Display:**  
   - Information about each cluster, including the number of files and average similarity, is displayed.
   - The number of outliers is also reported.

## Usage

To run the entire pipeline, execute the following command:

```bash
python llm_text_clustering.py
```

Make sure to place your PDF files in the `llm_text_clustering_data` directory before running the script.

## Results

The results include:

- **Thematic Descriptions:** A summary of the main topics and subtopics for each cluster, generated using GPT-4.
- **Cluster Visualization:** An interactive plot showing the clusters and their average similarity scores.
- **Cluster Information:** Details about the number of files in each cluster, average similarity, and the number of outliers.

### Example Output

**Thematic Descriptions:**  

- **Cluster 0:** "This cluster focuses on machine learning and artificial intelligence, with subtopics including neural networks, deep learning, and natural language processing."
- **Cluster 1:** "This cluster discusses climate change and environmental science, covering topics such as carbon emissions, renewable energy, and sustainability."

**Cluster Visualization:**  

- An interactive scatter plot showing clusters as circles, with the X-axis representing cluster IDs and the Y-axis representing average similarity scores.

**Cluster Information:**  

```
Number of outliers: 5
Cluster 0: 10 files, Average Similarity: 0.85
Cluster 1: 8 files, Average Similarity: 0.78
```

---

Developed to facilitate the automated and interactive organization of large volumes of documents into thematic groups.
