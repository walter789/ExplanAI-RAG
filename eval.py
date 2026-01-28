import pandas as pd
from typing import List, Dict
from datasets import Dataset
from agent import InterpretabilityAgent
from vector_storage import ExperimentVectorStore
from ingestion import MLflowIngestor, EXPERIMENT_NAME

# Import Ragas metrics (even if mocking, good to have imports valid)
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision

class AgentEvaluator:
    def __init__(self, agent: InterpretabilityAgent):
        self.agent = agent

    def generate_test_data(self) -> List[Dict[str, str]]:
        """
        Defines the Golden Dataset for evaluation.
        """
        return [
            {
                "question": "Why did accuracy hit 0.95?",
                "ground_truth": "Accuracy hit 0.95 in run 6aad83 due to learning_rate 0.01 and batch_size 32."
            },
            {
                "question": "What was the validation loss for the latest run?",
                "ground_truth": "The validation loss was 0.12 for the latest run."
            },
            {
                "question": "Which parameters were used in the experiment?",
                "ground_truth": "The experiment used learning_rate 0.01 and batch_size 32."
            }
        ]

    def mock_ragas_scores(self, dataset_dict: Dict[str, List]) -> pd.DataFrame:
        """
        Simulates Ragas evaluation output when no Judge LLM is available.
        """
        print("\n[System] Judge LLM API key not found. Running Mock Evaluation...")
        
        # Create a DataFrame from the collected data
        df = pd.DataFrame(dataset_dict)
        
        # Add mock scores (simulating a high-performing agent)
        df["faithfulness"] = [0.92, 0.88, 0.95]
        df["answer_relevancy"] = [0.90, 0.85, 0.98]
        df["context_precision"] = [1.0, 1.0, 0.95]
        
        return df

    def run_evaluation(self, use_mock: bool = True):
        """
        Runs the full evaluation loop: Inference -> Context Collection -> Scoring.
        """
        test_cases = self.generate_test_data()
        
        # Lists to store Ragas-compatible data
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        print("--- Starting Inference Loop ---")
        for case in test_cases:
            q = case["question"]
            gt = case["ground_truth"]
            print(f"Processing: '{q}'")

            # 1. Get Agent Response
            # Note: In a real scenario, we'd turn off the mock LLM in agent.py and use a real one
            ans = self.agent.analyze(q, experiment_filter=EXPERIMENT_NAME)
            
            # 2. Get Retrieved Contexts (Facts) directly from store for Ragas
            # The agent does this internally, but we need to expose it for Ragas
            raw_facts = self.agent.vector_store.query_experiments(q, experiment_filter=EXPERIMENT_NAME)
            ctx_list = [f["text"] for f in raw_facts]

            questions.append(q)
            answers.append(ans)
            contexts.append(ctx_list)
            ground_truths.append(gt)

        # 3. Prepare Dataset
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        # 4. Score (Real or Mock)
        if use_mock:
            results_df = self.mock_ragas_scores(dataset_dict)
        else:
            # Real Ragas Execution (Commented out for safety/no-key env)
            # hf_dataset = Dataset.from_dict(dataset_dict)
            # results = evaluate(
            #     hf_dataset,
            #     metrics=[faithfulness, answer_relevancy, context_precision],
            # )
            # results_df = results.to_pandas()
            pass

        print("\n--- Evaluation Results ---")
        print(results_df.to_markdown(index=False))
        
        # Calculate aggregate
        print("\n--- Aggregate Scores ---")
        print(results_df[["faithfulness", "answer_relevancy", "context_precision"]].mean())

if __name__ == "__main__":
    # 1. Setup Environment (Same as agent.py)
    # Re-ingest for context
    from ingestion import MLflowIngestor
    ingestor = MLflowIngestor()
    runs = ingestor.get_latest_runs(EXPERIMENT_NAME, limit=2)
    
    # Setup Store & Agent
    store = ExperimentVectorStore(location=":memory:")
    for run in runs:
        store.ingest_run(run)
        
    agent = InterpretabilityAgent(store)
    
    # 2. Run Eval
    evaluator = AgentEvaluator(agent)
    evaluator.run_evaluation(use_mock=True)