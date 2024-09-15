import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.linear_model import LogisticRegression
import random

# Set page configuration
st.set_page_config(page_title="Custom NLP Topic Modeling", layout="wide")

# Initialize session state variables if they don't exist
if 'labeled_indices' not in st.session_state:
    st.session_state.labeled_indices = []
if 'labels' not in st.session_state:
    st.session_state.labels = {}
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'batch_labeling' not in st.session_state:
    st.session_state.batch_labeling = False

# Helper function to cache the data loading
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Helper function to cache the creation of embeddings
@st.cache_resource
def create_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, show_progress_bar=True)

# Helper function to perform semantic search
def semantic_search(query, embeddings, df, num_results=10, page=0):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=num_results * (page + 1))
    indices = top_results.indices.tolist()[page * num_results:(page + 1) * num_results]
    return indices

# Load and display data
st.sidebar.header("Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.write(df.head())

    # Optional Sampling
    st.sidebar.header("Optional Sampling")
    sampling = st.sidebar.checkbox("Sample records?", value=False)
    if sampling:
        num_samples = st.sidebar.number_input("Number of records to sample", min_value=1, max_value=len(df), value=min(1000, len(df)))
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    # Select text column
    st.sidebar.header("Step 2: Select the Text Column")
    text_column = st.sidebar.selectbox("Select the text column", df.columns)

    # Create embeddings
    if st.session_state.embeddings is None:
        st.session_state.embeddings = create_embeddings(df[text_column].tolist())

    # Semantic Search
    st.sidebar.header("Step 3: Semantic Search for Initial Labeling")
    query = st.sidebar.text_input("Enter a search query:")
    num_results = st.sidebar.number_input("Number of search results to show per page", min_value=5, max_value=100, value=10)
    page = st.sidebar.number_input("Page number", min_value=0, value=0, step=1)

    if query:
        search_indices = semantic_search(query, st.session_state.embeddings, df, num_results=num_results, page=page)
        st.write("### Search Results")
        for idx in search_indices:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(df.iloc[idx][text_column])
                with col2:
                    label_selected = st.checkbox("Select as topic", key=f"select_{idx}")
                    if label_selected and idx not in st.session_state.labeled_indices:
                        st.session_state.labeled_indices.append(idx)
                        st.session_state.labels[idx] = 1
                    elif not label_selected and idx in st.session_state.labeled_indices:
                        st.session_state.labeled_indices.remove(idx)
                        del st.session_state.labels[idx]

    # Active Learning Batch Labeling
    if st.sidebar.button("Start Batch Labeling"):
        st.session_state.batch_labeling = True

    if st.session_state.batch_labeling:
        st.sidebar.header("Active Learning Batch Labeling")

        # Function to select data points for labeling based on uncertainty/random sampling
        def get_labeling_indices(model, embeddings, labeled_indices, num_to_label=10, proportion_random=0.5):
            uncertain_indices = uncertainty_sampling(model, embeddings, labeled_indices)
            random_indices = random.sample(list(set(range(len(df))) - set(labeled_indices)), num_to_label)
            final_indices = uncertain_indices[:int(num_to_label * (1 - proportion_random))] + random_indices[:int(num_to_label * proportion_random)]
            return final_indices

        def uncertainty_sampling(model, embeddings, labeled_indices):
            # Dummy uncertainty sampling, can be expanded based on the actual model
            if model:
                probs = model.predict_proba(embeddings)
                uncertainties = 1 - np.max(probs, axis=1)
                return np.argsort(uncertainties).tolist()
            return []

        # Get batch of samples to label
        if st.sidebar.button("Fetch Next Batch for Labeling"):
            batch_size = st.sidebar.slider("Batch size", min_value=5, max_value=50, value=10)
            labeling_indices = get_labeling_indices(st.session_state.model, st.session_state.embeddings, st.session_state.labeled_indices, num_to_label=batch_size)

            st.write("### Label these records:")
            for idx in labeling_indices:
                with st.container():
                    st.write(df.iloc[idx][text_column])
                    label = st.radio(f"Label for index {idx}", ["Yes", "No"], key=f"label_{idx}")
                    if label == "Yes":
                        st.session_state.labels[idx] = 1
                    elif label == "No":
                        st.session_state.labels[idx] = 0

        if st.sidebar.button("Submit and Retrain"):
            # Perform model training once a batch is completed
            labeled_indices = list(st.session_state.labels.keys())
            X_train = [st.session_state.embeddings[i] for i in labeled_indices]
            y_train = [st.session_state.labels[i] for i in labeled_indices]

            st.session_state.model = LogisticRegression(max_iter=1000)
            st.session_state.model.fit(X_train, y_train)

            st.write("Model retrained with current labels.")

    # Display overall progress and metrics
    st.sidebar.header("Progress and Metrics")
    total_labeled = len(st.session_state.labels)
    positive_labeled = sum(st.session_state.labels.values())
    st.sidebar.metric("Total Labeled", total_labeled)
    st.sidebar.metric("Positive Labels", positive_labeled)

else:
    st.write("Please upload a CSV file to begin.")
