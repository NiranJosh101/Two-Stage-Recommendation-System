from kfp import dsl, compiler

from src.utils.logging import logging


@dsl.container_component
def train_two_tower_op(
    feature_store_uri: str,
    config_path: str,
    checkpoint_path: dsl.Output[dsl.Model],
):
    return dsl.ContainerSpec(
    
        image='docker.io/josh305/two_tower_training_component:latest', 
        command=['python', '-m', 'src.two_tower_training.components.train_two_tower.two_tower_component'],
        args=[
            '--feature-store-uri', feature_store_uri,
            '--config', config_path,
            '--checkpoint-path', checkpoint_path.path, 
        ]
    )

logging.info("Defining Two-Tower Training Pipeline")
@dsl.pipeline(
    name="job-retrieval-training-pipeline",
    description="Trains a Two-Tower model for Job-User matching."
)
def two_tower_pipeline(
    
    feature_store_uri: str = "gs://your-actual-bucket/features/v1",
    config_file: str = "/app/src/two_tower_training/src_retriever/two_tower_config/config.yaml",
):
    train_task = train_two_tower_op(
        feature_store_uri=feature_store_uri,
        config_path=config_file
    )
    logging.info("Two-Tower Training Task added to the pipeline")
    
    train_task.set_display_name("Two-Tower Training Step")

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=two_tower_pipeline,
        package_path='two_tower_pipeline.yaml'
    )