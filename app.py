import streamlit as st
import pandas as pd
from typing import List
from ingestion import MLflowIngestor, RunSnapshot
from vector_storage import ExperimentVectorStore
from agent import InterpretabilityAgent

# --- Page Config ---
st.set_page_config(
    page_title="ExplanAI-RAG",
    page_icon="🔍",
    layout="wide"
)

# --- Custom CSS for Cards ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .card h3 {
        color: #333;
        margin-top: 0;
        font-size: 1.2rem;
    }
    .card p {
        color: #555;
        font-size: 1rem;
        line-height: 1.6;
    }
    .warning-card {
        border-left: 5px solid #FF9800;
    }
    .action-card {
        border-left: 5px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management & Caching ---

@st.cache_resource
def get_vector_store():
    """Initializes the in-memory vector store."""
    return ExperimentVectorStore(location=":memory:")

@st.cache_resource
def get_agent(_vector_store):
    """Initializes the agent."""
    return InterpretabilityAgent(_vector_store)

def main():
    st.title("🔍 ExplanAI-RAG: ML Observability Assistant")
    st.markdown("Production-grade root cause analysis for ML experiments.")

    vector_store = get_vector_store()
    agent = get_agent(vector_store)

    # --- Sidebar: Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        tracking_uri = st.text_input("MLflow Tracking URI", value="file:./mlruns")
        
        if st.button("Connect & List Experiments"):
            try:
                ingestor = MLflowIngestor(tracking_uri)
                st.session_state['ingestor'] = ingestor
                st.success("Connected to MLflow!")
            except Exception as e:
                st.error(f"Connection failed: {e}")

        experiment_name = st.text_input("Experiment Name", value="ExplanAI_Demo_Experiment")
        
        st.divider()
        
        if st.button("🔄 Ingest/Refresh Data"):
            if 'ingestor' not in st.session_state:
                st.session_state['ingestor'] = MLflowIngestor(tracking_uri)
            
            ingestor = st.session_state['ingestor']
            
            with st.spinner(f"Ingesting latest runs from '{experiment_name}'..."):
                runs = ingestor.get_latest_runs(experiment_name, limit=5)
                if runs:
                    for run in runs:
                        vector_store.ingest_run(run)
                    st.success(f"Successfully indexed {len(runs)} runs into Qdrant!")
                else:
                    st.warning("No runs found for this experiment.")

    # --- Main Analysis Area ---
    
    query = st.text_area("Ask a question about your experiments:", 
                         placeholder="e.g., Why did the loss spike in the last run?",
                         height=100)

    if st.button("🚀 Run Analysis", type="primary"):
        if not query:
            st.warning("Please enter a question.")
            return

        with st.spinner("Analyzing artifacts and reasoning..."):
            response = agent.analyze(query, experiment_filter=experiment_name)
            
            facts = vector_store.query_experiments(query, experiment_filter=experiment_name)

        # --- Results Display ---
        
        parts = response.split("**")
        observation = "No specific observation generated."
        hypothesis = "No hypothesis generated."
        action = "No action generated."

        for i, part in enumerate(parts):
            if "Observation:" in part:
                observation = parts[i+1].strip()
            elif "Root Cause Hypothesis:" in part:
                hypothesis = parts[i+1].strip()
            elif "Recommended Action:" in part:
                action = parts[i+1].strip()

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""<div class="card">
                <h3>🧐 Observation</h3>
                <p>{observation}</p>
            </div>""", unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""<div class="card warning-card">
                <h3>📉 Root Cause</h3>
                <p>{hypothesis}</p>
            </div>""", unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""<div class="card action-card">
                <h3>🛠️ Recommended Action</h3>
                <p>{action}</p>
            </div>""", unsafe_allow_html=True)

        if response.startswith("Error:"):
            st.toast("⚠️ Numerical verification failed! Check logs.", icon="❌")
            st.error(response)
        else:
            st.toast("✅ Numerical claims verified against logs.", icon="🛡️")

        with st.expander("📚 Retrieved Context (Transparency Layer)"):
            if facts:
                for i, fact in enumerate(facts):
                    st.markdown(f"**Fact {i+1} (Run {fact['run_id']})**")
                    st.code(fact['text'])
                    st.caption(f"Relevance Score: {fact['score']:.4f}")
            else:
                st.info("No relevant facts found in the vector store.")

if __name__ == "__main__":
    main()