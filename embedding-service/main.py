import os
import logging
from src.model_loader import ModelLoader
from src.feature_reader import FeatureReader
from src.embedder import JobEmbedder
from src.vector_writer import VectorWriter

# Setup logging to track progress in production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_embedding_pipeline():
    # 1. Configuration (Usually pulled from environment variables)
    MODEL_NAME = "job_recommender_v1"
    MODEL_VERSION = "2"
    DATA_PATH = r"C:\Users\USER\Desktop\Two_stage_recommendation_system\rs_feature_repo\feature_repo\data\job_features_v1.parquet"
    VDB_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "job-embeddings"

    try:
        # 2. Initialization Phase
        logger.info("Initializing services...")
        
        # Load the Two-Tower model from MLflow
        model_wrapper = ModelLoader(MODEL_NAME, MODEL_VERSION)
        model = model_wrapper.get_model()
        
        # Initialize our custom modules
        reader = FeatureReader(source_path=DATA_PATH, batch_size=1024)
        embedder = JobEmbedder(model=model)
        writer = VectorWriter(api_key=VDB_API_KEY, index_name=INDEX_NAME)

        # 3. Execution Phase (The Loop)
        logger.info(f"Starting batch embedding from {DATA_PATH}...")
        
        for batch_count, batch in enumerate(reader.stream_batches()):
            # Step A: Generate 256-dim normalized embeddings
            job_ids, vectors = embedder.compute(batch)
            
            # Step B: Upsert to Vector DB
            # We can also pass metadata here if the reader provides it
            writer.upsert_batch(ids=job_ids, vectors=vectors)
            
            logger.info(f"Processed batch {batch_count + 1} ({len(job_ids)} jobs)")

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_embedding_pipeline() 