from typing import Dict, Any


JOBS_RAW_CONTRACT: Dict[str, Any] = {
    "dataset_name": "jsearch_jobs_raw",
    "primary_key": "job_id",

    "fields": {
        
        "job_id": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "job_title": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "job_description": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "employer_name": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "job_publisher": {
            "type": str,
            "required": True,
            "nullable": False,
        },

      
        "job_employment_type": {
            "type": str,
            "required": False,
            "nullable": True,
        },

        
        "job_location": {
            "type": str,
            "required": False,
            "nullable": True,
        },
        "job_city": {
            "type": str,
            "required": False,
            "nullable": True,
        },
        "job_state": {
            "type": str,
            "required": False,
            "nullable": True,
        },
        "job_country": {
            "type": str,
            "required": False,
            "nullable": True,
        },

       
        "job_is_remote": {
            "type": bool,
            "required": False,
            "nullable": True,
        },

       
        "job_min_salary": {
            "type": float,
            "required": False,
            "nullable": True,
        },
        "job_max_salary": {
            "type": float,
            "required": False,
            "nullable": True,
        },
    },
}
