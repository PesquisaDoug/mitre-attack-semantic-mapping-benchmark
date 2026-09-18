from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
EMBEDDINGS = ROOT / "artifacts" / "embeddings"

st.set_page_config(page_title="MITRE ATT&CK Transformer Mapper", layout="wide")
st.title("MITRE ATT&CK Transformer Semantic Mapper")
st.caption(
    "Research demonstrator using pretrained MiniLM semantic retrieval followed by "
    "pretrained Cross-Encoder reranking."
)

required = [
    DATA / "techniques.csv",
    RESULTS / "metrics.csv",
    RESULTS / "experiment_manifest.json",
    EMBEDDINGS / "technique_embeddings.npy",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    st.error("Required artifacts are missing. Run the benchmark first.\n\n" + "\n".join(missing))
    st.stop()

techniques = pd.read_csv(DATA / "techniques.csv")
metrics = pd.read_csv(RESULTS / "metrics.csv")
manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
candidate_embeddings = np.load(EMBEDDINGS / "technique_embeddings.npy")
candidate_texts = techniques["document_text"].tolist()
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_models():
    bi = SentenceTransformer(manifest["models"]["bi_encoder"]["model_id"], device=device)
    cross = CrossEncoder(manifest["models"]["cross_encoder"]["model_id"], device=device)
    return bi, cross


bi_encoder, cross_encoder = load_models()

tabs = st.tabs(["Semantic Mapping", "Benchmark", "Dataset", "Reproducibility"])

with tabs[0]:
    st.subheader("Map CTI text to ATT&CK techniques")
    query = st.text_area(
        "Threat/procedure description",
        height=180,
        placeholder="Example: The malware executes PowerShell commands to download and run additional payloads...",
    )
    top_n = st.slider("Results", min_value=3, max_value=10, value=5)

    if st.button("Map to ATT&CK", type="primary"):
        query = query.strip()
        if not query:
            st.warning("Enter a CTI/procedure description.")
        else:
            query_embedding = bi_encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            semantic_scores = candidate_embeddings @ query_embedding
            candidate_order = np.argsort(-semantic_scores)[: manifest["rerank_top_k"]]
            pairs = [(query, candidate_texts[index]) for index in candidate_order]
            rerank_scores = np.asarray(cross_encoder.predict(pairs, show_progress_bar=False)).reshape(-1)
            rerank_order = np.argsort(-rerank_scores)

            rows = []
            for local_rank in rerank_order[:top_n]:
                index = int(candidate_order[local_rank])
                technique = techniques.iloc[index]
                rows.append(
                    {
                        "ATT&CK ID": technique["attack_id"],
                        "Technique": technique["name"],
                        "Cross-Encoder score": float(rerank_scores[local_rank]),
                        "Semantic score": float(semantic_scores[index]),
                        "Description": technique["description"],
                        "URL": technique["url"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.info(
                "Mappings are model-generated retrieval/ranking outputs and require analyst validation."
            )

with tabs[1]:
    st.subheader("Benchmark results")
    st.dataframe(metrics, use_container_width=True)
    for filename in [
        "04_strict_retrieval_benchmark.png",
        "05_all_vs_strict_ndcg.png",
        "06_query_latency.png",
        "07_attention_top_tokens.png",
    ]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[2]:
    st.subheader("ATT&CK corpus")
    c1, c2, c3 = st.columns(3)
    c1.metric("Techniques/sub-techniques", len(techniques))
    c2.metric("Evaluation queries", manifest["query_subsets"]["all"])
    c3.metric("Strict queries", manifest["query_subsets"]["strict"])
    st.dataframe(
        techniques[["attack_id", "name", "is_subtechnique", "tactics", "platforms"]],
        use_container_width=True,
    )

with tabs[3]:
    st.subheader("Reproducibility")
    st.json(manifest)
