import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


from retriver_model_archi import TwoTowerModel
from schema_validation import TwoTowerValidator, TowerSchema
from two_tower.metrics import recall_at_k, mrr_at_k, ndcg_at_k



class TwoTowerTrainer:
    def __init__(
        self,
        model: nn.Module,
        user_schema: Any,
        job_schema: Any,
        train_dataloader: DataLoader,
        learning_rate: float = 1e-3,
        temperature: float = 1.0, # Usually 1.0 for BCE, but can be tuned
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
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # CHANGED: Binary Cross Entropy with Logits for 0/1 classification
        self.loss_fn = nn.BCEWithLogitsLoss()

        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]):
        user_batch = batch["user_embedding"].to(self.device)
        job_batch = batch["job_embedding"].to(self.device)
        self.user_schema.validate_batch(user_batch)
        self.job_schema.validate_batch(job_batch)
        return user_batch, job_batch

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(self.dataloader):
            user_features, job_features = self._prepare_batch(batch)
            # CHANGED: Load the explicit labels (1 for positive, 0 for negative)
            labels = batch["label"].to(self.device).float() 
            
            self.optimizer.zero_grad()

            u_emb = torch.nn.functional.normalize(self.model.user_tower(user_features), p=2, dim=1)
            j_emb = torch.nn.functional.normalize(self.model.job_tower(job_features), p=2, dim=1)

            # CHANGED: Element-wise dot product for the specific pairs in the batch
            # shape: [batch_size]
            logits = torch.sum(u_emb * j_emb, dim=1) * self.temperature

            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            mlflow.log_metric("batch_loss", loss.item(), step=(epoch * len(self.dataloader)) + batch_idx)
            
        return total_loss / len(self.dataloader)

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader, k: int = 10):
        """
        Evaluation still uses the 'Retrieval' logic (All-to-All) 
        to see how well the model ranks items in a real-world scenario.
        """
        self.model.eval()
        all_u, all_j, all_true = [], [], []
        for batch in val_dataloader:
            u_feat, j_feat = self._prepare_batch(batch)
            all_u.append(self.model.user_tower(u_feat).cpu())
            all_j.append(self.model.job_tower(j_feat).cpu())
            # true_item_indices should represent the positive job index for the user
            all_true.extend(batch["true_item_indices"])

        u_final, j_final = torch.cat(all_u), torch.cat(all_j)
        
        return {
            f"recall_at_{k}": recall_at_k(u_final, j_final, all_true, k),
            f"mrr_at_{k}": mrr_at_k(u_final, j_final, all_true, k),
            f"ndcg_at_{k}": ndcg_at_k(u_final, j_final, all_true, k)
        }

    def train(self, epochs: int = 5, val_dataloader: Optional[DataLoader] = None, k: int = 10, min_recall_threshold: float = 0.7):
        with mlflow.start_run() as run:
            mlflow.log_params({"epochs": epochs, "learning_rate": self.optimizer.param_groups[0]['lr'], "loss_type": "BCE_Pointwise"})
            best_recall = 0.0

            for epoch in range(epochs):
                avg_loss = self.train_epoch(epoch)
                mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

                if val_dataloader:
                    metrics = self.evaluate(val_dataloader, k)
                    mlflow.log_metrics(metrics, step=epoch)
                    
                    current_recall = metrics[f"recall_at_{k}"]
                    best_recall = max(best_recall, current_recall)

            # Register and Promote logic
            model_info = mlflow.pytorch.log_model(self.model, "model", registered_model_name=self.model_name)

            if best_recall >= min_recall_threshold:
                model_version = model_info.registered_model_version
                self.client.transition_model_version_stage(
                    name=self.model_name, version=model_version, stage="Production", archive_existing_versions=True
                )
                mlflow.set_tag("deployment_status", "promoted_to_production")