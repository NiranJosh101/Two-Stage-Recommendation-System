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

import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")

load_dotenv() 




def run_embedding_pipeline():
    try:
        
        config_manager = ConfigurationManager()
        
        
        ml_cfg = config_manager.get_mlflow_config()
        data_cfg = config_manager.get_data_config()
        pc_cfg = config_manager.get_pinecone_config()

        logging.info("Initializing services with dynamic config...")
        
        
        manager = IndexManager(api_key=pc_cfg.api_key, index_name=pc_cfg.index_name)
        manager.ensure_index_exists(dimension=pc_cfg.dimension, metric=pc_cfg.metric)

    
        import mlflow
        mlflow.set_tracking_uri(ml_cfg.tracking_uri)
        
        model_wrapper = ModelLoader(ml_cfg.model_name, ml_cfg.model_version)
        model = model_wrapper.get_model()
        
       
        reader = FeatureReader(source_path=str(data_cfg.source_path), batch_size=data_cfg.batch_size)
        embedder = JobEmbedder(model=model)
        writer = VectorWriter(api_key=pc_cfg.api_key, index_name=pc_cfg.index_name)

      
        logging.info(f"Starting batch embedding from {data_cfg.source_path}...")
        for batch_count, batch in enumerate(reader.stream_batches()):
            job_ids, vectors = embedder.compute(batch)
            writer.upsert_batch(ids=job_ids, vectors=vectors)
            logging.info(f"Processed batch {batch_count + 1} ({len(job_ids)} jobs)")
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error("An error occurred during the embedding pipeline execution.")
        raise RecommendationsystemDataServie(e, sys) from e

if __name__ == "__main__":
    run_embedding_pipeline()