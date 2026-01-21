from kfp import dsl, compiler

@dsl.container_component
def train_two_tower_op(
    feature_store_uri: str,
    config_path: str,
    # dsl.Output[dsl.Model] is the correct way to handle model artifacts
    checkpoint_path: dsl.Output[dsl.Model],
):
    return dsl.ContainerSpec(
        # UPDATE THIS: Use your actual pushed image URI
        image='docker.io/josh305/two_tower_training_component:latest', 
        command=['python', '-m', 'src.two_tower_training.components.train_two_tower.two_tower_component'],
        args=[
            '--feature-store-uri', feature_store_uri,
            '--config', config_path,
            # Kubeflow will pass the local path where it expects the model to be written
            '--checkpoint-path', checkpoint_path.path, 
        ]
    )

@dsl.pipeline(
    name="job-retrieval-training-pipeline",
    description="Trains a Two-Tower model for Job-User matching."
)
def two_tower_pipeline(
    # 1. Update this to your actual GCS/S3 bucket
    feature_store_uri: str = "gs://your-actual-bucket/features/v1",
    
    # 2. IMPORTANT: If this file is INSIDE the image, keep the absolute path.
    # If it is in GCS, change this to a gs:// path.
    config_file: str = "/app/src/two_tower_training/src_retriever/two_tower_config/config.yaml",
):
    train_task = train_two_tower_op(
        feature_store_uri=feature_store_uri,
        config_path=config_file
    )
    
    # Resource allocation (Uncomment if using Vertex AI / Cloud)
    # train_task.set_cpu_limit('4').set_memory_limit('16G')
    
    train_task.set_display_name("Two-Tower Training Step")

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=two_tower_pipeline,
        package_path='two_tower_pipeline.yaml'
    )