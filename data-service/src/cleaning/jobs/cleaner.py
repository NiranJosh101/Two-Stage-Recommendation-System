from typing import Dict, Any
from src.cleaning.jobs.normalizers import JobNormalizers


class JobCleaner:
    """
    Applies deterministic cleaning to jobs that have already passed validation.
    Input: validated job dict (JOBS_RAW_CONTRACT)
    Output: cleaned job dict (same schema, join-safe)
    """

    def __init__(self):
        self.normalizer = JobNormalizers()

    def clean(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single job record.
        """
        return {
            "job_id": job["job_id"],  

            "job_title": self.normalizer.normalize_string(job.get("job_title")),
            "job_description": self.normalizer.normalize_job_description(
                job.get("job_description")
            ),
            "employer_name": self.normalizer.normalize_string(
                job.get("employer_name")
            ),

            "job_employment_type": self.normalizer.normalize_employment_type(
                job.get("job_employment_type")
            ),

            "job_location": self.normalizer.normalize_string(job.get("job_location")),
            "job_city": self.normalizer.normalize_string(job.get("job_city")),
            "job_state": self.normalizer.normalize_string(job.get("job_state")),
            "job_country": self.normalizer.normalize_string(job.get("job_country")),

            "job_is_remote": self.normalizer.normalize_boolean(
                job.get("job_is_remote")
            ),

            "job_min_salary": self.normalizer.normalize_salary(
                job.get("job_min_salary")
            ),
            "job_max_salary": self.normalizer.normalize_salary(
                job.get("job_max_salary")
            ),
        }
    

    def clean_many(self, jobs):

        jobs_list = jobs.to_dict(orient="records")  # now each job is a dict
        return [self.clean(job) for job in jobs_list]

