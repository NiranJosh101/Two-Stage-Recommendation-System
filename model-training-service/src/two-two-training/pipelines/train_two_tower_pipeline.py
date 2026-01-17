from kfp import dsl
from kfp import compiler

# 1. Define the Component from the Docker Image
@dsl.container_component
def train_two_tower_op(
    train_path: str,
    val_path: str,
    config_path: str,
    checkpoint_path: dsl.OutputPath(str),
):
    return dsl.ContainerSpec(
        image='gcr.io/your-project/train_two_tower:latest', # Change to your registry
        command=['python', 'component.py'],
        args=[
            '--train-path', train_path,
            '--val-path', val_path,
            '--config', config_path,
            '--checkpoint-path', checkpoint_path,
        ]
    )

# 2. Define the Pipeline Logic
@dsl.pipeline(
    name="job-retrieval-training-pipeline",
    description="Trains a Two-Tower model for Job-User matching."
)
def two_tower_pipeline(
    train_data: str = "gs://your-bucket/data/train.parquet",
    val_data: str = "gs://your-bucket/data/val.parquet",
    config_file: str = "/app/configs/two_tower.yaml"
):
    # Step 1: Run Training
    # We can request GPUs directly in the pipeline definition
    train_task = train_two_tower_op(
        train_path=train_data,
        val_path=val_data,
        config_path=config_file
    ).set_accelerator_type('nvidia-tesla-t4') \
     .set_gpu_limit(1) \
     .set_display_name("Train Two-Tower Model")

    # Step 2: (Optional) You could add an 'Upload to Vertex/Model Registry' step here
    # model_upload_task = ...

# 3. Compile the pipeline
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=two_tower_pipeline,
        package_path='two_tower_pipeline.yaml'
    )