from kfp import dsl, compiler

@dsl.container_component
def train_two_tower_op(
    feature_store_uri: str,
    config_path: str,
    # Using dsl.OutputPath ensures Kubeflow creates a managed path for your model
    checkpoint_path: dsl.Output[dsl.Model],
):
    return dsl.ContainerSpec(
        image='gcr.io/your-project/train_two_tower:latest',
        # Fixed: Use -m and the full module path to resolve the 'src' import issues
        command=['python', '-m', 'src.two_tower_training.components.train_two_tower.two_tower_component'],
        args=[
            '--feature-store-uri', feature_store_uri,
            '--config', config_path,
            '--checkpoint-path', checkpoint_path,
        ]
    )

@dsl.pipeline(
    name="job-retrieval-training-pipeline",
    description="Trains a Two-Tower model for Job-User matching."
)
def two_tower_pipeline(
    # Set these to your actual defaults or GCS paths
    feature_store_uri: str = "gs://your-bucket/features/v1",
    config_file: str = "src/two_tower_training/src_retriever/two_tower_config/config.yaml",
    # gpu_type: str = "nvidia-tesla-t4",
    # gpu_limit: int = 1
):
    train_task = train_two_tower_op(
        feature_store_uri=feature_store_uri,
        config_path=config_file
    )
    
    # Resource allocation for GPU
    # train_task.set_accelerator_type(gpu_type)
    # train_task.set_gpu_limit(gpu_limit)
    train_task.set_display_name("Two-Tower Training Step")

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=two_tower_pipeline,
        package_path='two_tower_pipeline.yaml'
    )