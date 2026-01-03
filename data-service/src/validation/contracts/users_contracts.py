from typing import Dict, Any

USERS_RAW_CONTRACT: Dict[str, Any] = {
    "dataset_name": "users_20251231_145354",
    "primary_key": "user_id",

    "fields": {

        "user_id": {
            "type": str,
            "required": True,
            "nullable": False,
        },

        "primary_roles": {
            "type": list,
            "required": True,
            "nullable": False,
        },
        "skills": {
            "type": list,
            "required": True,
            "nullable": False,
        },
        "experience_level": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "education_level": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "location": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "years_of_experience": {
            "type": float,
            "required": True,
            "nullable": False,
        },
    },
}
