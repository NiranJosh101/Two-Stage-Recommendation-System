from kfp import dsl
from kfp import compiler

@dsl.container_component
def train_two_tower_op(
    feature_store_uri: str,  # Updated to match your main script logic
    config_path: str,
    checkpoint_path: dsl.OutputPath=None
):
    return dsl.ContainerSpec(
        image='gcr.io/your-project/train_two_tower:latest',
        command=['python', 'train.py'], # Points to your training script
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
    feature_store_uri: str = "gs://your-bucket/features/v1",
    config_file: str = "two_tower_config/config.yaml",
    gpu_type: str = "nvidia-tesla-t4",
    gpu_limit: int = 1
):
    # The pipeline parameters now drive the resource allocation
    train_task = train_two_tower_op(
        feature_store_uri=feature_store_uri,
        config_path=config_file
    ).set_accelerator_type(gpu_type) \
     .set_gpu_limit(gpu_limit) \
     .set_display_name("Two-Tower Training Step")

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=two_tower_pipeline,
        package_path='two_tower_pipeline.yaml'
    )