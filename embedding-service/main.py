import os
import logging
import sys
from venv import logger
from dotenv import load_dotenv

from src.model_loader import ModelLoader
from src.feature_reader import FeatureReader
from src.embedder import JobEmbedder
from src.vector_writer import VectorWriter
from src.pc_embeds_index import IndexManager 

from src.config.config_manager import ConfigurationManager

from src.utils.logging import logging
from src.utils.exception import RecommendationsystemDataServie




load_dotenv() 




def run_embedding_pipeline():
    try:
        # 1. Initialize Configuration
        config_manager = ConfigurationManager()
        
        ml_cfg = config_manager.get_mlflow_config()
        data_cfg = config_manager.get_data_config()
        pc_cfg = config_manager.get_pinecone_config()

        # 2. GLOBAL MLFLOW SETUP (Must happen before ModelLoader)
        import mlflow
        mlflow.set_tracking_uri(ml_cfg.tracking_uri)
        logging.info(f"MLflow Tracking URI set to: {ml_cfg.tracking_uri}")

        logging.info("Initializing services with dynamic config...")
        
        # 3. Initialize Index Manager
        manager = IndexManager(api_key=pc_cfg.api_key, index_name=pc_cfg.index_name)
        manager.ensure_index_exists(dimension=pc_cfg.dimension, metric=pc_cfg.metric)
        
        # 4. Initialize Model Wrapper with config values
        # This will now correctly use 'job_recommender_v1' and '2'
        model_wrapper = ModelLoader(
            model_name=ml_cfg.model_name, 
            stage_or_version=ml_cfg.model_version
        )
        model = model_wrapper.get_model()
        
        # 5. Initialize Components
        reader = FeatureReader(source_path=str(data_cfg.source_path), batch_size=data_cfg.batch_size)
        embedder = JobEmbedder(model=model)
        writer = VectorWriter(api_key=pc_cfg.api_key, index_name=pc_cfg.index_name)

        # 6. Run Pipeline
        logging.info(f"Starting batch embedding from {data_cfg.source_path}...")
        for batch_count, batch in enumerate(reader.stream_batches()):
            job_ids, vectors = embedder.compute(batch)
            writer.upsert_batch(ids=job_ids, vectors=vectors)
            logging.info(f"Processed batch {batch_count + 1} ({len(job_ids)} jobs)")
            
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise RecommendationsystemDataServie(e, sys) from e
if __name__ == "__main__":
    run_embedding_pipeline()