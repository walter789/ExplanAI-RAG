# ExplanAI-RAG: ML Observability

ExplanAI-RAG is a specialized Retrieval-Augmented Generation (RAG) system designed to bridge the gap between MLOps artifacts and natural language reasoning. Unlike generic RAG implementations, this system is engineered specifically to ingest, index, and reason over structured ML experiment data (metrics, hyperparameters, and logs).

It serves as an autonomous "ML Lead" agent, capable of diagnosing training instabilities, explaining metric anomalies, and recommending architectural pivots based on historical experiment context.

## 🏗 System Architecture

The architecture follows a strict ETL-RAG pattern designed for data consistency and retrieval precision:

1.  **Ingestion Layer (`ingestion.py`)**: 
    -   Connects to **MLflow Tracking Servers**.
    -   Extracts raw run artifacts (metrics, params, start times).
    -   Enforces data quality using **Pydantic V2** schemas (`RunSnapshot`), ensuring only valid, typed data enters the system.

2.  **Vector Store & Indexing (`vector_storage.py`)**:
    -   Converts structured schemas into "Contextual Chunks".
    -   Generates embeddings using `all-MiniLM-L6-v2` (optimized for dense retrieval).
    -   Stores vectors in **Qdrant**, utilizing payload-based storage for rich metadata (Run IDs, Experiment Names).

3.  **Reasoning Engine (`agent.py`)**:
    -   Orchestrates the retrieve-then-generate workflow.
    -   Constructs grounded prompts with strict constraints.
    -   Applies post-generation verification logic.

## 🧠 The "Sentences of Fact" Strategy

Standard RAG approaches often fail with tabular data because embedding models (like BERT-based transformers) struggle to interpret raw JSON or CSV rows without semantic context.

ExplanAI-RAG implements a **Contextual Density** transformation. Instead of embedding `{ "accuracy": 0.95 }`, the system transforms this into a natural language declarative statement:
> *"In Run [ID] for Experiment 'ExplanAI_Demo', the accuracy was 0.95 and the learning_rate was set to 0.01."*

**Why this matters:**
-   **Semantic Alignment:** This format aligns the data with the pre-training objective of the embedding model (Sentence-BERT).
-   **High-Dimensional Clustering:** It ensures that queries like *"Why did the model converge?"* semantically cluster near runs with high accuracy and specific hyperparameter combinations.

## 🛡️ Numerical Safety Guardrails

A critical failure mode in Financial and Scientific RAG systems is **Numerical Hallucination**—where an LLM accurately retrieves the context but generates a slightly different number in the response.

ExplanAI-RAG implements a deterministic **Regex-Verification Layer** (`verify_accuracy_claims`):
1.  Extracts all numerical claims regarding key metrics (e.g., Accuracy) from the LLM's generated response.
2.  Cross-references these values against the exact float values present in the retrieved Qdrant payloads.
3.  **Hard Stop:** If a discrepancy is detected (beyond floating-point tolerance), the system rejects the response and flags an error, preventing misleading insights from reaching the engineer.

## 🔍 Retrieval Logic & Metadata Isolation

To prevent "Cross-Experiment Contamination"—where facts from *Project A* bleed into the reasoning for *Project B*—the system leverages Qdrant's **Metadata Filtering**.

Every query is implicitly filtered by `experiment_name`. This ensures that the Semantic Search (ANN) only operates within the manifold of the relevant experiment, strictly enforcing context isolation at the database level before the LLM ever sees the data.

## 📊 Performance Evaluation (Ragas)

Trust in autonomous agents must be quantified. This project integrates the **Ragas** framework to benchmark the agent's performance across three key dimensions:

1.  **Faithfulness:** Measures if the answer is derived *solely* from the retrieved facts (preventing external hallucinations).
2.  **Answer Relevancy:** Quantifies how directly the response addresses the user's specific root-cause question.
3.  **Context Precision:** Evaluates the signal-to-noise ratio of the retrieved log chunks.

## 📂 Core Project Structure

```plaintext
ExplanAI-RAG/
├── agent.py            # Reasoning Engine: Prompt construction & Guardrails
├── app.py              # UI Layer: Streamlit Dashboard for interaction
├── eval.py             # Benchmarking: Ragas evaluation loop & test cases
├── ingestion.py        # ETL Layer: MLflow connection & Pydantic schemas
├── vector_storage.py   # Database Layer: Qdrant client & Embedding generation
└── requirements.txt    # Dependency specifications
```
