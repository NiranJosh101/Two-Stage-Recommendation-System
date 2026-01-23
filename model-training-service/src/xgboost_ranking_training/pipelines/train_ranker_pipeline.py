from kfp import dsl
from kfp import compiler

# 1. Define the component (referencing the logic we just built)
@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas", 
        "xgboost", 
        "mlflow", 
        "pyarrow", 
        "scikit-learn"
    ]
)
def train_ranker_op(
    feature_store_uri: str,
    config_path: str,
    model_output_path: dsl.Output[dsl.Model]
):

    # from ranking_training_module import run_ranking_training
    import argparse

    args = argparse.Namespace(
        config=config_path,
        feature_store_uri=feature_store_uri,
        model_output_path=model_output_path
    )
    
    # run_ranking_training(args)

# 2. Define the Pipeline
@dsl.pipeline(
    name="XGBoost Ranking Training Pipeline",
    description="Orchestrates data prep, training, and evaluation for the Job Ranker."
)
def ranking_pipeline(
    feature_store_uri: str,
    config_path: str = "config/ranking_config.yaml"
):
    # Training Task
    training_task = train_ranker_op(
        feature_store_uri=feature_store_uri,
        config_path=config_path
    )

    # You can add post-processing or deployment steps here
    # training_task.set_display_name("XGBoost Ranker Training")
    # training_task.set_cpu_limit('4')
    # training_task.set_memory_limit('8G')

# 3. Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ranking_pipeline,
        package_path="ranking_pipeline.yaml"
    )