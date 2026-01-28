import re
import os
from typing import List, Dict, Any, Optional
from vector_storage import ExperimentVectorStore
from ingestion import EXPERIMENT_NAME

class InterpretabilityAgent:
    """
    The reasoning engine that connects User Questions -> Vector Retrieval -> LLM Analysis.
    """
    def __init__(self, vector_store: ExperimentVectorStore):
        self.vector_store = vector_store

    def build_prompt(self, query: str, facts: List[Dict[str, Any]]) -> str:
        """
        Constructs a grounded prompt for the LLM.
        """
        context_str = "\n".join([f"{i+1}. {f['text']}" for i, f in enumerate(facts)])
        
        prompt = f"""
You are an expert Senior ML Architect and Lead Engineer.
Your task is to analyze ML experiment logs and answer the user's question.

### STRICT CONSTRAINTS:
1. Answer ONLY using the provided facts below. Do not hallucinate external information.
2. If the answer isn't in the context, say "Data not found in logs."
3. Format your response exactly as follows:
   **Observation:** [What do the logs say?]
   **Root Cause Hypothesis:** [Why might this be happening based on the params/metrics?]
   **Recommended Action:** [What specific steps should the engineer take?]

### AVAILABLE CONTEXT (FACTS):
{context_str}

### USER QUESTION:
{query}
"""
        return prompt.strip()

    def call_llm(self, prompt: str) -> str:
        """
        Execute the LLM call. 
        """
        # In a production environment, this should connect to OpenAI, Anthropic, or a local model.
        # Returning a default analysis for demonstration purposes.
        return """
**Observation:** The logs indicate that the accuracy reached 0.9500 and the loss was 0.1200 in the reviewed run.

**Root Cause Hypothesis:** The low loss (0.1200) combined with high accuracy (0.9500) suggests the model converged successfully. Rapid convergence might imply the task is simpler than anticipated or potential data leakage if the test set wasn't properly isolated.

**Recommended Action:** Validate the train-test split to ensure no data leakage. If the split is valid, consider benchmarking against a simpler baseline model or increasing task complexity.
"""

    def verify_accuracy_claims(self, response: str, facts: List[Dict[str, Any]]) -> bool:
        """
        Self-Correction: Checks if 'Accuracy' numbers in the response match the retrieved facts.
        """
        fact_accuracies = []
        for fact in facts:
            match = re.search(r"accuracy was (\d+\.\d+)", fact['text'])
            if match:
                fact_accuracies.append(float(match.group(1)))
        
        if not fact_accuracies:
            return True

        response_accuracies = re.findall(r"accuracy.*?(\d+\.\d+)", response.lower())
        
        if not response_accuracies:
            return True

        for acc_str in response_accuracies:
            val = float(acc_str)
            if not any(abs(val - fa) < 1e-4 for fa in fact_accuracies):
                return False
        
        return True

    def analyze(self, query: str, experiment_filter: Optional[str] = None) -> str:
        """
        Main pipeline: Retrieve -> Prompt -> Generate -> Verify.
        """
        facts = self.vector_store.query_experiments(query, experiment_filter=experiment_filter)
        
        if not facts:
            return "Data not found in logs."

        prompt = self.build_prompt(query, facts)
        
        response = self.call_llm(prompt)
        
        if not self.verify_accuracy_claims(response, facts):
            return "Error: The generated response failed fact-checking (numerical mismatch). Please check raw logs."
            
        return response

if __name__ == "__main__":
    store = ExperimentVectorStore(location=":memory:")
    
    from ingestion import MLflowIngestor
    ingestor = MLflowIngestor()
    runs = ingestor.get_latest_runs(EXPERIMENT_NAME, limit=2)
    for run in runs:
        store.ingest_run(run)
        
    agent = InterpretabilityAgent(store)
    
    user_query = "Why is the model loss so low and what should I check next?"
    print(f"\n--- User Query: {user_query} ---")
    
    final_answer = agent.analyze(user_query, experiment_filter=EXPERIMENT_NAME)
    
    print("\n--- Agent Response ---")
    print(final_answer)