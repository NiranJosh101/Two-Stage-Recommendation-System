import os
import logging
from dotenv import load_dotenv

from src.model_loader import ModelLoader
from src.feature_reader import FeatureReader
from src.embedder import JobEmbedder
from src.vector_writer import VectorWriter
from src.pc_embeds_index import IndexManager 

import mlflow

# Tell MLflow to look at the server you started in the other terminal
mlflow.set_tracking_uri("http://127.0.0.1:5000")

load_dotenv() 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_embedding_pipeline():
    MODEL_NAME = "job_recommender_v1"
    MODEL_VERSION = "2"
    DATA_PATH = r"C:\Users\USER\Desktop\Two_stage_recommendation_system\rs_feature_repo\feature_repo\data\job_features_v1.parquet"
    VDB_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "job-embeddings"

    try:
        logger.info("Initializing services...")
        
        # 1. INFRASTRUCTURE CHECK
        # We do this first so we don't waste time loading the model if the DB is down
        manager = IndexManager(api_key=VDB_API_KEY, index_name=INDEX_NAME)
        manager.ensure_index_exists(dimension=256, metric="dotproduct")

        # 2. LOAD AI ASSETS
        model_wrapper = ModelLoader(MODEL_NAME, MODEL_VERSION)
        model = model_wrapper.get_model()
        
        # 3. SETUP DATA FLOW
        reader = FeatureReader(source_path=DATA_PATH, batch_size=1024)
        embedder = JobEmbedder(model=model)
        writer = VectorWriter(api_key=VDB_API_KEY, index_name=INDEX_NAME)

        # 4. EXECUTION
        logger.info(f"Starting batch embedding from {DATA_PATH}...")
        for batch_count, batch in enumerate(reader.stream_batches()):
            job_ids, vectors = embedder.compute(batch)
            writer.upsert_batch(ids=job_ids, vectors=vectors)
            logger.info(f"Processed batch {batch_count + 1} ({len(job_ids)} jobs)")

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_embedding_pipeline()