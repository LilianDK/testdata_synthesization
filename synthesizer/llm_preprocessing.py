from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from synthesizer.config import co

co = co

print("START TEXT SPLITTING")
text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
loader = PyPDFLoader(
    "./data/FuerArbeitSonstiges_Abdul et al_2018_Trends and trajectories for explainable.pdf"
)
raw_chunks = loader.load_and_split(text_splitter)
print(raw_chunks[1])
content = [rc.page_content for rc in raw_chunks]
print(f"END TEXT SPLITTING n = {len(content)}")

source = [rc.metadata["source"] for rc in raw_chunks]
page = [rc.metadata["page"] for rc in raw_chunks]
content = [rc.page_content for rc in raw_chunks]

df = pd.DataFrame({"name": source, "page": page, "text_chunks": content})
# df.to_excel("results.xlsx")


def cluster_embeddings(embeddings, num_clusters) -> list:
    """Clustering with vector embeddings

    Args:
        embeddings (float): Vector embedding
        num_clusters (int): Defined number of clusters k

    Returns:
        list: List of associated cluster per embedding
    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.predict(embeddings)
    # cluster_labels = kmeans.labels_
    return cluster_labels


# ------------------------------------------------------------- EMBEDDING
# Embedding of texts
print("START TEXT EMBEDDING")
embeddings = []
model = "embed-english-light-v3.0"

for index, row in df.iterrows():
    text_to_embed = row["text_chunks"]
    array = [text_to_embed]
    # Generate text using the text-generation model based on the text to embed
    response = co.embed(array, model=model, input_type="clustering")
    embedding = response.embeddings[0]
    embeddings.append(embedding)
    # print(f"Original Text: {text_to_embed}")
    # print(f"Generated Embedding: {embedding}")

embeddings = np.array(embeddings)
print(type(embeddings))
print("END TEXT EMBEDDING")

# ------------------------------------------------------------- IDENTIFY K
# Initialize kmeans parameters
print("START IDENTIFY K")
kmeans_kwargs = {
    "init": "random",
    "n_init": 100,
    "random_state": 1,
}

# Calculate SSE for given k range
sse = []
start = 1
end = 20
x = np.arange(start, end)

for k in range(start, end):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)

# Visualize k range
kneedle = KneeLocator(x=x, y=sse, S=1.0, curve="convex", direction="decreasing")
print(kneedle.knee)
print(kneedle.elbow)
print(round(kneedle.knee_y, 3))

kneedle_plot1 = kneedle.plot_knee_normalized()
kneedle_plot2 = kneedle.plot_knee()
print("END IDENTIFY K")
print(f"THE OPTIMAL K IS k={kneedle.knee}")
# ------------------------------------------------------------- CLUSTERING
# Clustering of embeddings
print("START DOCUMENT CLUSTERING")
k = kneedle.knee
cluster_labels = cluster_embeddings(embeddings, k)
df_cluster_labels = pd.DataFrame(cluster_labels)
df = pd.DataFrame(df)
df_combined = pd.concat([df, df_cluster_labels], ignore_index=False, axis=1)

# Export results
df_combined.to_excel(f"results/first_stage_results.xlsx")
df_combined.head()
print("END DOCUMENT CLUSTERING")
