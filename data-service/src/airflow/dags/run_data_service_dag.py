from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "data-platform",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="recommendation_system_data_service_v1",
    description="Full orchestration from Raw Ingestion to Feast Feature Store",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["two stage recommendation-system", "feast", "ingestion"],
) as dag:

    
    ingest_jobs = BashOperator(
        task_id="ingest_jobs",
        bash_command="python -m src.ingestion.jobs.run_jobs_ingestion",
    )
    ingest_users = BashOperator(
        task_id="ingest_users",
        bash_command="python -m src.ingestion.users.run_users_ingestion",
    )
    ingest_interactions = BashOperator(
        task_id="ingest_interactions",
        bash_command="python -m src.ingestion.user_interactions.run_user_interaction",
    )

    
    validate_raw = BashOperator(
        task_id="validate_raw_data",
        bash_command="python -m src.validation.run_validation",
    )

  
    clean_jobs = BashOperator(
        task_id="clean_jobs",
        bash_command="python -m src.cleaning.jobs.run_jobs_cleaning",
    )
    clean_users = BashOperator(
        task_id="clean_users",
        bash_command="python -m src.cleaning.users.run_user_cleaner",
    )
    clean_interactions = BashOperator(
        task_id="clean_interactions",
        bash_command="python -m src.cleaning.interactions.run_interactions_cleaner",
    )

    
    feature_transform = BashOperator(
        task_id="feature_transform",
        bash_command="python -m src.cleaning.feature_transform.run_transform",
    )


    general_labeling = BashOperator(
        task_id="general_labeling",
        bash_command="python -m src.supervision.labeling.run_labeling",
    )
    positive_labeling = BashOperator(
        task_id="positive_labeling",
        bash_command="python -m src.supervision.labeling.run_positive_labels",
    )
    negative_sampling = BashOperator(
        task_id="negative_sampling",
        bash_command="python -m src.supervision.negative_sampling.run_negative_sampling",
    )


    build_pos_neg_samples = BashOperator(
        task_id="assemble_samples",
        bash_command="python -m src.supervision.assembly.run_build_dataset",
    )
    build_two_tower_ds = BashOperator(
        task_id="build_two_tower_dataset",
        bash_command="python -m src.supervision.assembly.final_dataset_assembly.run_feature_build",
    )
    build_ranking_ds = BashOperator(
        task_id="build_ranking_dataset",
        bash_command="python -m src.supervision.assembly.run_rankingds_build",
    )

   
    publish_to_feast = BashOperator(
        task_id="publish_to_feast",
        bash_command="python -m src.feature_store.run_fs_write",
    )

   

   
    [ingest_jobs, ingest_users, ingest_interactions] >> validate_raw


    validate_raw >> [clean_jobs, clean_users, clean_interactions]

    
    [clean_jobs, clean_users] >> feature_transform

    
    [feature_transform, clean_interactions] >> general_labeling
    general_labeling >> [positive_labeling, negative_sampling]


    [positive_labeling, negative_sampling] >> build_pos_neg_samples
    build_pos_neg_samples >> [build_two_tower_ds, build_ranking_ds]

   
    [build_two_tower_ds, build_ranking_ds] >> publish_to_feast