import os
import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError, ConfigDict
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

class RunSnapshot(BaseModel):
    """
    Structured representation of an MLflow Run.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    run_id: str = Field(..., description="Unique identifier for the MLflow run")
    experiment_id: str = Field(..., description="ID of the experiment this run belongs to")
    experiment_name: str = Field(..., description="Name of the experiment")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Key-value pairs of logged metrics")
    params: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs of logged parameters")
    timestamp: datetime.datetime = Field(..., description="Timestamp of the run start")

def to_rag_chunks(snapshot: RunSnapshot) -> List[str]:
    """
    Transforms a RunSnapshot into a list of descriptive strings optimized for semantic retrieval.
    """
    chunks = []
    
    base_ctx = f"In Run {snapshot.run_id} for Experiment '{snapshot.experiment_name}' at {snapshot.timestamp.isoformat()}"
    
    if snapshot.metrics:
        metrics_str = ", ".join([f"{k} was {v:.4f}" for k, v in snapshot.metrics.items()])
        chunks.append(f"{base_ctx}, the recorded metrics were: {metrics_str}.")
        
    if snapshot.params:
        params_str = ", ".join([f"{k} was set to {v}" for k, v in snapshot.params.items()])
        chunks.append(f"{base_ctx}, the configuration parameters were: {params_str}.")
        
    if not chunks:
        chunks.append(f"{base_ctx}, no metrics or parameters were logged.")
        
    return chunks

class MLflowIngestor:
    """
    Handles connection to MLflow and retrieval of experiment artifacts.
    """
    def __init__(self, tracking_uri: Optional[str] = None):
        try:
            if not tracking_uri:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                tracking_uri = f"file:///{os.path.join(base_dir, 'mlruns').replace('\\', '/')}"
            
            self.tracking_uri = tracking_uri
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            print(f"INFO: Connected to MLflow at {self.tracking_uri}")
        except Exception as e:
            print(f"ERROR: Failed to initialize MLflow client: {e}")
            raise

    def get_latest_runs(self, experiment_name: str, limit: int = 5) -> List[RunSnapshot]:
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                available = [e.name for e in self.client.search_experiments()]
                print(f"WARNING: Experiment '{experiment_name}' not found. Available: {available}")
                return []

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=limit,
                order_by=["attribute.start_time DESC"]
            )

            snapshots = []
            for run in runs:
                try:
                    start_time = datetime.datetime.fromtimestamp(run.info.start_time / 1000.0)
                    
                    snapshot = RunSnapshot(
                        run_id=run.info.run_id,
                        experiment_id=run.info.experiment_id,
                        experiment_name=experiment_name,
                        metrics=run.data.metrics,
                        params=run.data.params,
                        timestamp=start_time
                    )
                    snapshots.append(snapshot)
                except ValidationError as ve:
                    print(f"ERROR: Validation failed for run {run.info.run_id}: {ve}")
                    continue
            
            return snapshots

        except Exception as e:
            print(f"ERROR: Failed to fetch runs for experiment '{experiment_name}': {e}")
            return []

EXPERIMENT_NAME = "ExplanAI_Demo_Experiment"

if __name__ == "__main__":
    ingestor = MLflowIngestor()
    
    print("--- Setting up Demo Data ---")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 32)
        mlflow.log_metric("validation_loss", 0.45)
        mlflow.log_metric("accuracy", 0.88)
        print(f"Created demo run: {run.info.run_id}")

    print("\n--- Starting Ingestion ---")
    runs = ingestor.get_latest_runs(EXPERIMENT_NAME, limit=2)
    
    print(f"\nFound {len(runs)} run(s). Processing into RAG chunks...\n")
    
    for run in runs:
        chunks = to_rag_chunks(run)
        for chunk in chunks:
            print(f"- {chunk}")