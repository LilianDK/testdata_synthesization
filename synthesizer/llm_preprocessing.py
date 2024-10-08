import json
import random
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
from dotenv import load_dotenv
from kneed import KneeLocator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sklearn.cluster import KMeans

from synthesizer.config import CHUNK_OVERLAP, CHUNK_SIZE, co, embedding_model, input_type, llm_model, number_of_texts

directory = "./data/"
load_dotenv(override=True)

np.random.seed(123)

def preprocessing() -> None:
    # Initialize logging for the text splitting process
    logger.info("START TEXT SPLITTING")

    # Create a text splitter object using a recursive character splitter or nltk splitter
    text_splitter = RecursiveCharacterTextSplitter().from_tiktoken_encoder( model_name="gpt-4",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP)

    # Check if the input data type is JSON
    if input_type == "JSON":
        # Initialize lists to store various attributes of the documents
        document_id = []
        source_url = []
        text = []
        filenames = []

        # Iterate over all JSON files in the specified directory
        for name in Path(directory).rglob("*.json"):
        # Open and read the JSON file with UTF-8 encoding
        # https://docs.astral.sh/ruff/rules/os-path-join/
            with name.open(mode="r", encoding="utf-8") as file:

                # Load the JSON content into a dictionary
                data_dict = json.load(file)

            # Initialize a counter for invalid data
            i = 0
            # Iterate over each item in the dictionary
            for item in data_dict:
                try:
                    # Extract relevant information from the JSON structure
                    doc_id = item["document_id"]
                    source = item["metadata"]["source"]["url"]
                    t = item["text"]
                    filename = item["extracted_metadata"]["filename"]
                except KeyError:
                    # Increment the invalid data counter if a KeyError occurs
                    i += 1
                # Append the extracted information to the respective lists
                document_id.append(doc_id)
                filenames.append(filename)
                source_url.append(source)
                text.append([t[0].replace("\n", " ")])
            # Log the amount of invalid data encountered
            logger.info(f"Amount of Invalid Data {i}")

        # Create a DataFrame from the extracted information and save it as a CSV file
        df = pd.DataFrame({"document_id": document_id, "filename": filenames, "source_url": source_url, "text": text})
        df.to_csv("results/data_overview.csv", encoding="utf-8")

        # Randomly sample 5 texts from the list of texts
        text = random.sample(text, number_of_texts)

        # Initialize lists to store the split texts and their associated document IDs and source URLs
        text_splits = []
        document_idlist = []
        source_urllist = []
        # Iterate over the sampled texts
        for idx, t in enumerate(text):
            # Split the text using the text splitter
            splits = text_splitter.split_text(t[0].replace("\n", " "))
            # Append the split texts and their associated information to the respective lists
            for s in splits:
                text_splits.append(s)
                document_idlist.append(document_id[idx])
                source_urllist.append(source_url[idx])

        # Create a DataFrame from the split texts and their associated information
        df = pd.DataFrame({"document_id": document_idlist, "source_url": source_urllist, "text_chunks": text_splits})

    # Check if the input data type is PDF
    if input_type == "PDF":
        # Initialize a list to store chunks of raw text
        raw_chunks = []

        # Iterate over all PDF files in the specified directory
        for name in Path(directory).rglob("*.pdf"):
            # Load and split the PDF file using the text splitter
            loader = PyPDFLoader(name)
            extraction = loader.load_and_split(text_splitter)
            raw_chunks.append(extraction)

        # Initialize lists to store the source, page number, and content of each chunk
        source =[]
        page = []
        content = []
        # Iterate over the raw chunks
        for r in raw_chunks:
            for x in r:
                # Append the source, page number, and content to the respective lists
                source.append(x.metadata["source"])
                page.append(x.metadata["page"])
                content.append(x.page_content)

        # Create a DataFrame from the extracted information
        df = pd.DataFrame({"name": source, "page": page, "text_chunks": content})

    # ------------------------------------------------------------- EMBEDDING
    # Start the process of embedding text data
    logger.info("START TEXT EMBEDDING")
    embeddings = []

    # Iterate over each row in the dataframe to process the text data
    for _index, row in df.iterrows():
        text_to_embed = row["text_chunks"][0]# Extract the text to be embedded
        array = [text_to_embed] # Create an array with the text

        # Depending on the chosen language model, generate the text embedding
        if llm_model == "cohere":
            # Use the Cohere model to generate embeddings for clustering
            response = co.embed(texts=array, model=embedding_model, input_type="clustering")
            embedding = response.embeddings[0]
        if llm_model == "ollama":
            # Use the Ollama model to generate embeddings
            response = ollama.embeddings(prompt=array[0], model=embedding_model)
            embedding = response["embedding"]
        embeddings.append(embedding)

    # Convert the list of embeddings to a NumPy array for further processing
    embeddings = np.array(embeddings)

    # ------------------------------------------------------------- IDENTIFY K
    # Begin the process of identifying the optimal number of clusters (K)
    logger.info("START IDENTIFY K")
    # Set the initial parameters for the KMeans clustering algorithm
    kmeans_kwargs = {
        "init": "random",
        "n_init": 100,
        "random_state": 1,
    }

    # Prepare to calculate the sum of squared errors (SSE) for a range of K values
    sse = []
    start = 1
    end = 20
    x = np.arange(start, end) # Generate an array of K values

    # Calculate SSE for each K value in the range
    #for k in range(start, end):
    #    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    #    kmeans.fit(embeddings)
    #    sse.append(kmeans.inertia_)

    # Use the KneeLocator to find the "elbow" in the SSE curve, which indicates the optimal K
    #kneedle = KneeLocator(x=x, y=sse, S=1.0, curve="convex", direction="decreasing")
    #logger.info(kneedle.knee)
    #logger.info(kneedle.elbow)
    #logger.info(round(kneedle.knee_y, 3))

    # Uncomment the following lines if you want to visualize the knee/elbow plots
    # kneedle_plot1 = kneedle.plot_knee_normalized()
    # kneedle_plot2 = kneedle.plot_knee()
    #logger.info("END IDENTIFY K")
    #logger.info(f"THE OPTIMAL K IS k={kneedle.knee}") # Log the optimal K value
    # ------------------------------------------------------------- CLUSTERING
    # Begin the clustering of the text embeddings
    logger.info("START DOCUMENT CLUSTERING")
    #k = kneedle.knee # Use the optimal K value identified earlier
    k = 2
    cluster_labels = cluster_embeddings(embeddings, k) # Cluster the embeddings
    df_cluster_labels = pd.DataFrame(cluster_labels, columns=["cluster"]) # Create a dataframe with cluster labels
    df = pd.DataFrame(df) # Ensure df is a dataframe
    df_combined = pd.concat([df, df_cluster_labels], ignore_index=False, axis=1)

    df_combined["text_chunks"] = df_combined["text_chunks"].replace("\\n", " ")
    df_combined["text_chunks"] = df_combined["text_chunks"].apply(lambda x: [x])

    # Uncomment the following lines if you want to process the text chunks and cluster them
    #df_combined["text_cluster"] = df_combined.groupby("cluster")["text_chunks"].transform(lambda x: " ".join(x).replace("\n", " ").replace("\\n", " "))
    #df_combined = df_combined.drop_duplicates(subset=["cluster", "text_cluster"])
    #df_combined["text_cluster"] = df_combined["text_cluster"].apply(lambda x: [x]) # !!!!!
    #df_combined = df_combined.drop("text_chunks", axis=1)
    # Export the clustering results to a CSV file
    df_combined.to_csv("results/first_stage_results.csv", encoding="utf-8")
    logger.info("END DOCUMENT CLUSTERING") # End of the clustering process


def cluster_embeddings(embeddings: list[float], num_clusters: int) -> list:
    """Clustering with vector embeddings.

    Args:
    ----
        embeddings (float): Vector embedding
        num_clusters (int): Defined number of clusters k

    Returns:
    -------
        list: List of associated cluster per embedding

    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    return kmeans.predict(embeddings)
    # cluster_labels = kmeans.labels_
