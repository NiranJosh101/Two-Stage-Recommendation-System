import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient




from src.two_tower_training.src_retriever.two_tower.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.utils.exception import RecommendationsystemDataServie
from src.eval.evaluator import ModelPromoter
from src.utils.logging import logging


class TwoTowerTrainer:
    def __init__(
        self,
        model: nn.Module,
        user_schema: Any,
        job_schema: Any,
        train_dataloader: DataLoader,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        device: str = "cpu",
        checkpoint_path: str = "./checkpoints/two_tower.pt",
        experiment_name: str = "Job_Retrieval_TwoTower",
        model_name: str = "Job_Retrieval_Model"
    ):
        self.model = model.to(device)
        self.user_schema = user_schema
        self.job_schema = job_schema
        self.dataloader = train_dataloader
        self.device = device
        self.temperature = temperature
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()

        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

        print(
            f"[INIT] TwoTowerTrainer initialized | "
            f"device={device} | lr={learning_rate} | "
            f"temperature={temperature}"
        )

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]):
        try:
            """Prepare a batch of data for training. This includes moving the data to the appropriate device and validating it against the defined schemas."""
            user_batch = batch["user_embedding"].to(self.device)
            job_batch = batch["job_embedding"].to(self.device)

            self.user_schema.validate_batch(user_batch)
            self.job_schema.validate_batch(job_batch)

            return user_batch, job_batch
        except RecommendationsystemDataServie as e:
            logging.info(f"Schema validation failed: {e}")
            raise

    def train_epoch(self, epoch: int):
        try:
            self.model.train()
            total_loss = 0.0

            print(f"\n[TRAIN] Starting epoch {epoch + 1}")

            for batch_idx, batch in enumerate(self.dataloader):
                user_features, job_features = self._prepare_batch(batch)
                labels = batch["label"].to(self.device).float()

                self.optimizer.zero_grad()

                u_emb = torch.nn.functional.normalize(
                    self.model.user_tower(user_features), p=2, dim=1
                )
                j_emb = torch.nn.functional.normalize(
                    self.model.job_tower(job_features), p=2, dim=1
                )

                logits = torch.sum(u_emb * j_emb, dim=1) * self.temperature
                loss = self.loss_fn(logits, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            
                if batch_idx % 50 == 0:
                    print(
                        f"[TRAIN][Epoch {epoch + 1}] "
                        f"Batch {batch_idx}/{len(self.dataloader)} | "
                        f"Batch Loss: {loss.item():.4f}"
                    )

                mlflow.log_metric(
                    "batch_loss",
                    loss.item(),
                    step=(epoch * len(self.dataloader)) + batch_idx
                )

            avg_loss = total_loss / len(self.dataloader)
            print(
                f"[TRAIN] Epoch {epoch + 1} completed | "
                f"Avg Loss: {avg_loss:.4f}"
            )

            return avg_loss
        
        except RecommendationsystemDataServie as e:
            logging.error(f"Training halted due to schema validation error: {e}")
            raise

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader, k: int = 10):
        try:
            print(f"\n[EVAL] Running evaluation (Recall@{k}, MRR@{k}, NDCG@{k})")

            self.model.eval()
            all_u, all_j, all_true = [], [], []

            for batch in val_dataloader:
                u_feat, j_feat = self._prepare_batch(batch)

                all_u.append(self.model.user_tower(u_feat).cpu())
                all_j.append(self.model.job_tower(j_feat).cpu())
                all_true.extend(batch["true_item_indices"])

            u_final, j_final = torch.cat(all_u), torch.cat(all_j)

            metrics = {
                f"recall_at_{k}": recall_at_k(u_final, j_final, all_true, k),
                f"mrr_at_{k}": mrr_at_k(u_final, j_final, all_true, k),
                f"ndcg_at_{k}": ndcg_at_k(u_final, j_final, all_true, k),  
            }
            logging.info(f"Recall@{k}: {metrics[f'recall_at_{k}']:.4f}, MRR@{k}: {metrics[f'mrr_at_{k}']:.4f}, NDCG@{k}: {metrics[f'ndcg_at_{k}']:.4f}"),

            print(
                "[EVAL RESULTS] "
                + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            )

            return metrics
        
        except RecommendationsystemDataServie as e:
            logging.error(f"Evaluation halted due to schema validation error: {e}")
            raise


    def train(
        self,
        epochs: int = 5,
        val_dataloader: Optional[DataLoader] = None,
        k: int = 10,
        improvement_margin: float = 0.01
    ):
        try:
            print(
                f"\n[RUN] Starting training for {epochs} epochs | "
                f"improvement_margin={improvement_margin}"
            )

            with mlflow.start_run() as run:
                mlflow.log_params({
                    "epochs": epochs,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "loss_type": "BCE_Pointwise",
                    "improvement_margin": improvement_margin
                })

                best_recall = 0.0

                for epoch in range(epochs):
                    avg_loss = self.train_epoch(epoch)
                    mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

                    if val_dataloader:
                        metrics = self.evaluate(val_dataloader, k)
                        mlflow.log_metrics(metrics, step=epoch)

                        current_recall = metrics[f"recall_at_{k}"]
                        best_recall = max(best_recall, current_recall)

                        print(
                            f"[TRACKING] Best Recall@{k} so far: "
                            f"{best_recall:.4f}"
                        )

                print("\n[MODEL] Logging model to MLflow registry...")
                model_info = mlflow.pytorch.log_model(
                    self.model,
                    "model",
                    registered_model_name=self.model_name
                )

                promoter = ModelPromoter(model_name=self.model_name)
                champion_metrics = promoter.get_champion_metrics()
                
                # Define the dynamic threshold
                # If no champion exists, we use a low baseline (e.g., 0.1) to bootstrap
                champion_recall = champion_metrics.get(f"recall_at_{k}", 0.1) if champion_metrics else 0.1
                dynamic_threshold = champion_recall + improvement_margin

                print(f"\n[GATE] Champion Recall@{k}: {champion_recall:.4f}")
                print(f"[GATE] Target for Promotion: {dynamic_threshold:.4f}")

                # Log the model artifact
                model_info = mlflow.pytorch.log_model(
                    self.model, "model", registered_model_name=self.model_name
                )

                # The Promotion Decision
                if best_recall >= dynamic_threshold:
                    promoter.transition_to_production(model_info.registered_model_version)
                    mlflow.set_tag("deployment_status", "promoted_to_production")
                    print(f" SUCCESS: Challenger ({best_recall:.4f}) beat Champion. Promoted.")
                    return True # Signal to the Orchestrator that Retriever is ready
                else:
                    print(f" FAIL: Challenger ({best_recall:.4f}) did not meet margin.")
                    return False
                    
        except RecommendationsystemDataServie as e:
            logging.error(f"Training process halted due to schema validation error: {e}")
            raise
