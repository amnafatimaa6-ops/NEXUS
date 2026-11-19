
import os
import io
import zipfile
import requests
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import networkx as nx
import json

DATA_DIR = "nexus_data"

GITHUB_ZIP_URL = (
    "https://raw.githubusercontent.com/"
    "amnafatimaa6-ops/NEXUS/main/NEXUS_COMPLETE.zip"
)

@st.cache_resource
def load_nexus():

    os.makedirs(DATA_DIR, exist_ok=True)

    required = [
        "knowledge_base.csv",
        "timeline.csv",
        "character_intelligence.csv",
        "nexus_faiss.index",
        "knowledge_embeddings.npy",
        "character_network.graphml",
        "config.json"
    ]

    if not all(
        os.path.exists(os.path.join(DATA_DIR, f))
        for f in required
    ):

        response = requests.get(
            GITHUB_ZIP_URL,
            timeout=180
        )
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:

            for member in archive.namelist():

                filename = os.path.basename(member)

                if filename in required:
                    with archive.open(member) as source:
                        with open(
                            os.path.join(DATA_DIR, filename), "wb"
                        ) as target:
                            target.write(source.read())

    knowledge = pd.read_csv(
        os.path.join(DATA_DIR, "knowledge_base.csv")
    )

    timeline = pd.read_csv(
        os.path.join(DATA_DIR, "timeline.csv")
    )

    characters = pd.read_csv(
        os.path.join(DATA_DIR, "character_intelligence.csv")
    )

    index = faiss.read_index(
        os.path.join(DATA_DIR, "nexus_faiss.index")
    )

    embeddings = np.load(
        os.path.join(DATA_DIR, "knowledge_embeddings.npy")
    )

    graph = nx.read_graphml(
        os.path.join(DATA_DIR, "character_network.graphml")
    )

    with open(os.path.join(DATA_DIR, "config.json")) as f:
        config = json.load(f)

    return knowledge, timeline, characters, index, embeddings, graph, config
