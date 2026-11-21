
import os
import ast
import zipfile
import requests
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import faiss

from sentence_transformers import SentenceTransformer


# =========================================================
# NEXUS | MARVEL INTELLIGENCE ENGINE
# =========================================================

st.set_page_config(
    page_title="NEXUS | Marvel Intelligence",
    page_icon="🟥",
    layout="wide",
    initial_sidebar_state="expanded"
)

GITHUB_ZIP_URL = (
    "https://raw.githubusercontent.com/"
    "amnafatimaa6-ops/NEXUS/main/NEXUS_COMPLETE.zip"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "nexus_data"
ZIP_PATH = BASE_DIR / "NEXUS_COMPLETE.zip"


# =========================================================
# MARVEL-INSPIRED INTERFACE
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Roboto:wght@300;400;500;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 85% 5%, #351014 0%, transparent 30%),
        linear-gradient(135deg, #080808 0%, #111111 55%, #19090b 100%);
    color: #f4f4f4;
}

[data-testid="stSidebar"] {
    background: #100c0d;
    border-right: 1px solid #67151c;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1px;
}

h1 {
    color: #f12b36 !important;
}

h2, h3 {
    color: #eeeeee !important;
}

p, label, span, div {
    font-family: 'Roboto', sans-serif;
}

.stButton > button {
    background: linear-gradient(90deg, #9e101a, #e32636);
    color: white;
    border: 1px solid #ff4652;
    border-radius: 7px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #ed2636;
    border-color: white;
    color: white;
    transform: translateY(-2px);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1b1718, #251012);
    border: 1px solid #692027;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricValue"] {
    color: #ff3545;
}

.stTextInput input {
    background: #171313;
    color: white;
    border: 1px solid #8d2029;
}

hr {
    border-color: #5b1b21;
}

.nexus-banner {
    background: linear-gradient(100deg, #27090d, #111111 75%);
    border: 1px solid #8d1d28;
    border-left: 5px solid #e32636;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 25px;
}

.nexus-kicker {
    color: #ff4652;
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    letter-spacing: 4px;
}

.nexus-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin: 8px 0;
}

.nexus-subtitle {
    color: #b8aeb0;
    font-size: 14px;
}

.small-note {
    color: #a9a0a1;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DOWNLOAD AND EXTRACT DATA
# =========================================================

def download_and_extract():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        "knowledge_base.csv",
        "character_intelligence.csv",
        "timeline.csv",
        "nexus_faiss.index",
        "character_network.graphml"
    ]

    existing = list(DATA_DIR.rglob("knowledge_base.csv"))

    if existing:
        folder = existing[0].parent

        if all((folder / file).exists() for file in required_files):
            return folder

    st.info("Downloading NEXUS intelligence archive...")

    response = requests.get(
        GITHUB_ZIP_URL,
        stream=True,
        timeout=180
    )
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    st.info("Extracting intelligence archive...")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        # Prevent unsafe paths from escaping the data directory.
        root = DATA_DIR.resolve()

        for member in archive.infolist():
            target = (DATA_DIR / member.filename).resolve()

            if target != root and root not in target.parents:
                raise ValueError("Unsafe path found in ZIP archive.")

        archive.extractall(DATA_DIR)

    for knowledge_file in DATA_DIR.rglob("knowledge_base.csv"):

        folder = knowledge_file.parent

        if all((folder / file).exists() for file in required_files):
            return folder

    raise FileNotFoundError(
        "The ZIP was extracted, but one or more required NEXUS files "
        "could not be found. Check the archive contents."
    )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_resource(show_spinner="Loading NEXUS intelligence systems...")

def load_nexus():

    folder = download_and_extract()

    knowledge = pd.read_csv(folder / "knowledge_base.csv")
    characters = pd.read_csv(folder / "character_intelligence.csv")
    timeline = pd.read_csv(folder / "timeline.csv")

    index = faiss.read_index(
        str(folder / "nexus_faiss.index")
    )

    graph = nx.read_graphml(
        folder / "character_network.graphml"
    )

    return knowledge, characters, timeline, index, graph


try:

    (
        knowledge,
        characters,
        timeline,
        faiss_index,
        character_graph
    ) = load_nexus()

except Exception as error:

    st.error("NEXUS could not initialise.")
    st.exception(error)
    st.stop()


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource(show_spinner="Initialising semantic search...")

def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


try:
    embedding_model = load_embedding_model()

except Exception as error:
    embedding_model = None
    st.warning(
        "Semantic search is unavailable. "
        "Character profiles and graph intelligence can still work."
    )


# =========================================================
# CHARACTER ALIASES
# =========================================================

ALIASES = {

    "iron man": "IRON MAN/TONY STARK",
    "tony stark": "IRON MAN/TONY STARK",

    "spider-man": "SPIDER-MAN/PETER PARKER",
    "spiderman": "SPIDER-MAN/PETER PARKER",
    "peter parker": "SPIDER-MAN/PETER PARKER",

    "captain america": "CAPTAIN AMERICA",
    "steve rogers": "CAPTAIN AMERICA",

    "thor": "THOR/DR. DONALD BLAK",

    "hulk": "HULK/DR. ROBERT BRUC",
    "bruce banner": "HULK/DR. ROBERT BRUC",

    "scarlet witch": "SCARLET WITCH/WANDA",
    "wanda": "SCARLET WITCH/WANDA",

    "wolverine": "WOLVERINE/LOGAN",

    "black widow": "BLACK WIDOW/NATASHA"
}


def resolve_character(query):

    query = str(query).strip().lower()

    if query in ALIASES:
        return ALIASES[query]

    exact = characters[
        characters["character"].str.lower() == query
    ]

    if not exact.empty:
        return exact.iloc[0]["character"]

    matches = characters[
        characters["character"].str.lower().str.contains(
            query,
            regex=False,
            na=False
        )
    ]

    if not matches.empty:
        return matches.sort_values(
            "connections",
            ascending=False
        ).iloc[0]["character"]

    return None


# =========================================================
# CHARACTER PROFILE
# =========================================================

def get_profile(name):

    canonical = resolve_character(name)

    if canonical is None:
        return None

    result = characters[
        characters["character"] == canonical
    ]

    if result.empty:
        return None

    return result.iloc[0]


# =========================================================
# SEMANTIC KNOWLEDGE SEARCH
# =========================================================

def knowledge_search(query, top_k=5):

    if embedding_model is None:
        return pd.DataFrame()

    vector = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(vector)

    scores, indices = faiss_index.search(
        vector,
        top_k
    )

    valid = [
        idx for idx in indices[0]
        if 0 <= idx < len(knowledge)
    ]

    if not valid:
        return pd.DataFrame()

    results = knowledge.iloc[valid].copy()

    score_map = {
        idx: float(score)
        for idx, score in zip(indices[0], scores[0])
        if idx in valid
    }

    results["similarity"] = [
        score_map[idx] for idx in valid
    ]

    return results.sort_values(
        "similarity",
        ascending=False
    )


# =========================================================
# RELATIONSHIP EVIDENCE
# =========================================================

def get_relationships(character, top_k=10):

    results = knowledge[
        (knowledge["character"] == character) &
        (knowledge["document_type"] == "co_appearance")
    ].copy()

    if not results.empty:
        return results.head(top_k)

    # Fallback: graph neighbours
    if character in character_graph:

        neighbours = list(
            character_graph.neighbors(character)
        )

        return pd.DataFrame({
            "character": [character] * min(top_k, len(neighbours)),
            "document_type": ["graph_connection"] * min(top_k, len(neighbours)),
            "text": [
                f"{character} is connected to {name} in the character network."
                for name in neighbours[:top_k]
            ],
            "evidence": ["Character network graph"] * min(top_k, len(neighbours))
        })

    return pd.DataFrame()


# =========================================================
# NETWORK VISUALISATION
# =========================================================

def draw_character_network(character, limit=20):

    if character not in character_graph:
        st.warning("Character not found in the network graph.")
        return

    neighbours = list(
        character_graph.neighbors(character)
    )[:limit]

    selected = [character] + neighbours

    subgraph = character_graph.subgraph(selected)

    if len(subgraph) < 2:
        st.info("Not enough connections to display a network.")
        return

    positions = nx.spring_layout(
        subgraph,
        seed=42,
        k=0.8
    )

    edge_x = []
    edge_y = []

    for source, target in subgraph.edges():

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#7b3038"),
        hoverinfo="none"
    )

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []

    for node in subgraph.nodes():

        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        if node == character:
            node_sizes.append(28)
            node_colors.append("#ff2638")
        else:
            node_sizes.append(15)
            node_colors.append("#a92b36")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        textfont=dict(
            color="white",
            size=9
        ),
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(
                width=1,
                color="#ffffff"
            )
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace]
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor="#100d0e",
        plot_bgcolor="#100d0e",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TIMELINE DATA
# =========================================================

def parse_participants(value):

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []

    except Exception:
        return []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 0;">
        <div style="font-family:Orbitron; color:#ed2939;
                    font-size:30px; font-weight:800;">
            NEXUS
        </div>
        <div style="color:#aaa; letter-spacing:3px; font-size:10px;">
            MARVEL INTELLIGENCE ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Command Centre",
            "Character Intelligence",
            "Relationship Explorer",
            "Network Visualisation",
            "Timeline Intelligence",
            "Knowledge Search"
        ]
    )

    st.divider()

    st.caption("SYSTEM STATUS")

    st.success("Knowledge base loaded")
    st.success("FAISS index loaded")
    st.success("Character graph loaded")

    if embedding_model is not None:
        st.success("Semantic engine online")
    else:
        st.warning("Semantic engine offline")

    st.caption("NEXUS VERSION 1.0")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="nexus-banner">
    <div class="nexus-kicker">MARVEL UNIVERSE // INTELLIGENCE DIVISION</div>
    <div class="nexus-title">NEXUS</div>
    <div class="nexus-subtitle">
        Neural Exploration of Character Networks, Relationships
        and Unified Story Intelligence
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1: COMMAND CENTRE
# =========================================================

if page == "Command Centre":

    st.subheader("COMMAND CENTRE")

    st.write(
        "A unified intelligence dashboard for exploring "
        "Marvel character networks and comic appearances."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "CHARACTERS",
        f"{len(characters):,}"
    )

    col2.metric(
        "NETWORK NODES",
        f"{character_graph.number_of_nodes():,}"
    )

    col3.metric(
        "NETWORK EDGES",
        f"{character_graph.number_of_edges():,}"
    )

    col4.metric(
        "KNOWLEDGE DOCUMENTS",
        f"{len(knowledge):,}"
    )

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:

        st.subheader("Most Connected Characters")

        top = characters.sort_values(
            "connections",
            ascending=False
        ).head(10)

        fig = go.Figure(
            go.Bar(
                x=top["connections"],
                y=top["character"],
                orientation="h",
                marker_color="#e32636"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=470,
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="#100d0e",
            plot_bgcolor="#100d0e"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Most Influential Characters")

        top_influence = characters.sort_values(
            "pagerank",
            ascending=False
        ).head(10)

        st.dataframe(
            top_influence[
                ["character", "pagerank", "connections"]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Quick Character Lookup")

    quick_name = st.text_input(
        "Search the Marvel character database",
        placeholder="e.g. Iron Man, Thor, Spider-Man..."
    )

    if st.button("ACCESS CHARACTER"):

        profile = get_profile(quick_name)

        if profile is None:
            st.error("Character not found.")

        else:
            st.session_state["selected_character"] = profile["character"]
            st.success(f"Character identified: {profile['character']}")


# =========================================================
# PAGE 2: CHARACTER INTELLIGENCE
# =========================================================

elif page == "Character Intelligence":

    st.subheader("CHARACTER INTELLIGENCE")

    search_name = st.text_input(
        "Enter character name",
        placeholder="Iron Man"
    )

    if search_name:

        profile = get_profile(search_name)

        if profile is None:

            st.error("No matching character found.")

        else:

            character = profile["character"]

            st.markdown(f"### {character}")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "CONNECTIONS",
                f"{int(profile['connections']):,}"
            )

            c2.metric(
                "COMIC APPEARANCES",
                f"{int(profile['comic_appearances']):,}"
            )

            c3.metric(
                "PAGERANK",
                f"{profile['pagerank']:.6f}"
            )

            c4.metric(
                "COMMUNITY",
                str(profile["community"])
            )

            st.divider()

            st.subheader("Co-appearance Evidence")

            relationships = get_relationships(character)

            if not relationships.empty:

                st.dataframe(
                    relationships[
                        ["text", "evidence"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info("No relationship evidence available.")

            st.caption(
                "Co-appearance means two characters occur in the same "
                "comic records. It does not prove friendship, rivalry "
                "or another canonical relationship."
            )


# =========================================================
# PAGE 3: RELATIONSHIP EXPLORER
# =========================================================

elif page == "Relationship Explorer":

    st.subheader("RELATIONSHIP EXPLORER")

    search_name = st.text_input(
        "Choose a character",
        placeholder="Iron Man"
    )

    if search_name:

        profile = get_profile(search_name)

        if profile is None:
            st.error("Character not found.")

        else:

            character = profile["character"]

            st.markdown(f"### CONNECTION ANALYSIS: {character}")

            relationships = get_relationships(
                character,
                top_k=15
            )

            if not relationships.empty:

                st.dataframe(
                    relationships,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info("No relationship evidence found.")


# =========================================================
# PAGE 4: NETWORK VISUALISATION
# =========================================================

elif page == "Network Visualisation":

    st.subheader("CHARACTER NETWORK")

    search_name = st.text_input(
        "Enter character",
        placeholder="Spider-Man"
    )

    if search_name:

        profile = get_profile(search_name)

        if profile is None:

            st.error("Character not found.")

        else:

            character = profile["character"]

            st.write(
                f"Visualising the local network around {character}."
            )

            draw_character_network(character)


# =========================================================
# PAGE 5: TIMELINE INTELLIGENCE
# =========================================================

elif page == "Timeline Intelligence":

    st.subheader("NARRATIVE TIMELINE")

    if timeline.empty:

        st.info("No timeline events are currently stored.")

    else:

        display_timeline = timeline.copy()

        if "event_order" in display_timeline.columns:

            display_timeline = display_timeline.sort_values(
                "event_order"
            )

        if "participants" in display_timeline.columns:

            display_timeline["participants"] = (
                display_timeline["participants"]
                .apply(parse_participants)
                .apply(lambda x: ", ".join(map(str, x)))
            )

        st.metric(
            "RECORDED EVENTS",
            len(display_timeline)
        )

        st.dataframe(
            display_timeline,
            use_container_width=True,
            hide_index=True
        )

        if "participants" in timeline.columns:

            names = sorted({
                person
                for value in timeline["participants"]
                for person in parse_participants(value)
            })

            selected = st.selectbox(
                "Filter by participant",
                ["All characters"] + names
            )

            if selected != "All characters":

                filtered = timeline[
                    timeline["participants"].apply(
                        lambda x: selected in parse_participants(x)
                    )
                ]

                st.dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True
                )

    st.caption(
        "The current timeline contains extracted sample narrative events. "
        "It is not a complete chronological Marvel canon."
    )


# =========================================================
# PAGE 6: KNOWLEDGE SEARCH
# =========================================================

elif page == "Knowledge Search":

    st.subheader("NEXUS KNOWLEDGE RETRIEVAL")

    st.write(
        "Search the indexed character profiles and relationship evidence."
    )

    query = st.text_input(
        "Ask NEXUS",
        placeholder="Which characters appear with Iron Man most often?"
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=15,
        value=5
    )

    if st.button("RUN INTELLIGENCE SEARCH"):

        if not query.strip():

            st.warning("Enter a search query.")

        elif embedding_model is None:

            st.error(
                "Semantic search is unavailable because the embedding "
                "model could not be loaded."
            )

        else:

            with st.spinner("Searching NEXUS knowledge index..."):

                results = knowledge_search(
                    query,
                    top_k=top_k
                )

            if results.empty:

                st.info("No matching evidence found.")

            else:

                for _, row in results.iterrows():

                    st.markdown("---")

                    st.markdown(
                        f"**{row['character']}**"
                    )

                    st.caption(
                        f"{row['document_type']} | "
                        f"Similarity: {row['similarity']:.3f}"
                    )

                    st.write(row["text"])

                    st.caption(
                        f"Evidence source: {row['evidence']}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <div style="font-family:Orbitron; color:#e32636;
                font-size:16px; letter-spacing:3px;">
        NEXUS INTELLIGENCE SYSTEM
    </div>
    <div style="color:#777; font-size:11px; margin-top:8px;">
        MARVEL CHARACTER NETWORK ANALYSIS • KNOWLEDGE RETRIEVAL
        • GRAPH INTELLIGENCE
    </div>
</div>
""", unsafe_allow_html=True)
