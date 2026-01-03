from typing import Dict, Any

INTERACTIONS_RAW_CONTRACT: Dict[str, Any] = {
    "dataset_name": "interactions_raw",
    "primary_key": "interaction_id", 

    "fields": {
        
        "interaction_id": {
            "type": str,
            "required": True,
            "nullable": False,
        },

       
        "user_id": {
            "type": str,
            "required": True,
            "nullable": False,
        },
        "job_id": {
            "type": str,
            "required": True,
            "nullable": False,
        },

   
        "event_type": {
            "type": str,
            "required": True,
            "nullable": False,
        },
    },
}
