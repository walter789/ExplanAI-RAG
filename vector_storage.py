import uuid
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from ingestion import RunSnapshot, to_rag_chunks, MLflowIngestor, EXPERIMENT_NAME
import mlflow

class ExperimentVectorStore:
    """
    Manages embedding generation and vector storage for MLflow run artifacts using Qdrant.
    """
    COLLECTION_NAME = "ml_experiment_facts"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_SIZE = 384

    def __init__(self, location: str = ":memory:"):
        """
        Initialize the Vector Store.
        
        Args:
            location: Qdrant location (':memory:' for RAM, or path for disk).
        """
        print(f"INFO: Initializing Vector Store with model '{self.EMBEDDING_MODEL}'...")
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL)
        self.client = QdrantClient(location=location)
        
        # Ensure collection exists (recreating for fresh demo state)
        self._initialize_collection()

    def _initialize_collection(self):
        """Creates the Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections()
        exists = any(c.name == self.COLLECTION_NAME for c in collections.collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            print(f"INFO: Created collection '{self.COLLECTION_NAME}'")
        else:
            print(f"INFO: Collection '{self.COLLECTION_NAME}' already exists")

    def ingest_run(self, run: RunSnapshot):
        """
        Converts a run into vector points and upserts them to Qdrant.
        
        Args:
            run: The Pydantic model of the MLflow run.
        """
        chunks = to_rag_chunks(run)
        if not chunks:
            print(f"WARNING: No chunks generated for Run {run.run_id}")
            return

        # Generate embeddings
        embeddings = self.encoder.encode(chunks)
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = {
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "timestamp": run.timestamp.isoformat(),
                "text": chunk
            }
            
            # Create a point
            point = models.PointStruct(
                id=str(uuid.uuid4()),  # Random UUID for point ID
                vector=vector.tolist(),
                payload=payload
            )
            points.append(point)

        # Upsert
        operation_info = self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )
        print(f"INFO: Upserted {len(points)} facts for Run {run.run_id}. Status: {operation_info.status}")

    def query_experiments(self, query_text: str, experiment_filter: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the vector database for relevant experiment facts.
        
        Args:
            query_text: The user's natural language question.
            experiment_filter: Optional experiment name to filter by.
            limit: Number of results to return.
            
        Returns:
            List of matches with score and payload.
        """
        query_vector = self.encoder.encode(query_text).tolist()
        
        # Build Filter
        q_filter = None
        if experiment_filter:
            q_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="experiment_name",
                        match=models.MatchValue(value=experiment_filter)
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=q_filter,
            limit=limit
        ).points
        
        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text"),
                "run_id": hit.payload.get("run_id"),
                "experiment": hit.payload.get("experiment_name")
            }
            for hit in results
        ]

if __name__ == "__main__":
    # 1. Setup Data Source
    print("--- 1. Fetching Data from MLflow ---")
    # Ensure we use the same URI as ingestion.py
    ingestor = MLflowIngestor() 
    
    # Create a fresh run if none exist (or just to be sure)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("loss", 0.12)
        mlflow.log_param("optimizer", "adam")
    
    runs = ingestor.get_latest_runs(EXPERIMENT_NAME, limit=5)
    print(f"Fetched {len(runs)} runs.")

    # 2. Vector Storage
    print("\n--- 2. Ingesting into Qdrant ---")
    vector_store = ExperimentVectorStore(location=":memory:") # In-memory for demo
    
    for run in runs:
        vector_store.ingest_run(run)

    # 3. Test Search
    print("\n--- 3. Testing Semantic Search ---")
    QUERY = "What was the accuracy?"
    print(f"Query: '{QUERY}'")
    
    results = vector_store.query_experiments(QUERY, experiment_filter=EXPERIMENT_NAME)
    
    for i, res in enumerate(results, 1):
        print(f"\nResult {i} (Score: {res['score']:.4f}):")
        print(f"  Fact: {res['text']}")
        print(f"  Run ID: {res['run_id']}")