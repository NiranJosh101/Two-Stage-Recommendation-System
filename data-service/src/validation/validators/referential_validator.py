import pandas as pd

class ReferentialIntegrityError(Exception):
    """Raised when referential integrity checks fail."""
    pass


def validate_referential_integrity(
    interactions_df: pd.DataFrame,
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    dataset_name: str = "interactions_raw"
) -> None:
    
    errors = []

    invalid_users = interactions_df.loc[
        ~interactions_df["user_id"].isin(users_df["user_id"])
    ]
    if not invalid_users.empty:
        errors.append(
            f"{len(invalid_users)} interactions reference missing users in {dataset_name}:\n{invalid_users}"
        )

    invalid_jobs = interactions_df.loc[
        ~interactions_df["job_id"].isin(jobs_df["job_id"])
    ]
    if not invalid_jobs.empty:
        errors.append(
            f"{len(invalid_jobs)} interactions reference missing jobs in {dataset_name}:\n{invalid_jobs}"
        )

    if errors:
        raise ReferentialIntegrityError("\n".join(errors))
