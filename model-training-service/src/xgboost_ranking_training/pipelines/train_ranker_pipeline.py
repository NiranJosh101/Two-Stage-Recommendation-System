from kfp import dsl
from kfp import compiler


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

   
    import argparse

    args = argparse.Namespace(
        config=config_path,
        feature_store_uri=feature_store_uri,
        model_output_path=model_output_path
    )
    
    


@dsl.pipeline(
    name="XGBoost Ranking Training Pipeline",
    description="Orchestrates data prep, training, and evaluation for the Job Ranker."
)
def ranking_pipeline(
    feature_store_uri: str,
    config_path: str = "config/ranking_config.yaml"
):

    training_task = train_ranker_op(
        feature_store_uri=feature_store_uri,
        config_path=config_path
    )

  

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ranking_pipeline,
        package_path="ranking_pipeline.yaml"
    )