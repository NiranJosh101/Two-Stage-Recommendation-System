import mlflow
from mlflow.tracking import MlflowClient

class ModelPromoter:
    def __init__(self, model_name: str):
        self.client = MlflowClient()
        self.model_name = model_name

    def get_champion_metrics(self):
        """Fetches metrics from the model currently in 'Production' stage."""
        try:
            # Find the version currently tagged as Production
            latest_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
            if not latest_versions:
                return None
            
            prod_version = latest_versions[0]
            run = self.client.get_run(prod_version.run_id)
            return run.data.metrics
        except Exception:
            return None

    def evaluate_and_promote(self, challenger_metrics: dict, metric_key: str, min_improvement: float = 0.01):
        """
        Decision Logic:
        1. If no Champion exists, promote Challenger (Bootstrap).
        2. If Challenger > (Champion + Improvement Margin), promote.
        """
        champion_metrics = self.get_champion_metrics()
        
        # 1. Bootstrap (First time training)
        if not champion_metrics:
            print(f"No Champion found for {self.model_name}. Promoting first model.")
            return True

        # 2. Comparison Logic
        challenger_val = challenger_metrics.get(metric_key, 0)
        champion_val = champion_metrics.get(metric_key, 0)

        print(f"Comparing {metric_key}: Challenger({challenger_val:.4f}) vs Champion({champion_val:.4f})")

        if challenger_val > (champion_val + min_improvement):
            print(" Challenger is significantly better. Promoting...")
            return True
        else:
            print(" Challenger did not beat Champion by the required margin.")
            return False

    def transition_to_production(self, version: str):
        """Actual MLflow transition logic."""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )