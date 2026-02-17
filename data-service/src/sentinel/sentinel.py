import sqlite3
import requests
import os

def check_stockpile_and_trigger(new_rows_count: int, threshold: int = 10000):
    """
    Checks if the number of new rows in Feast exceeds the threshold.
    If so, it triggers the GitHub Action CT pipeline.
    """
    conn = sqlite3.connect("metadata/ingestion_stats.db")
    cursor = conn.cursor()
    
    # 1. Update the local counter
    cursor.execute("UPDATE stockpile SET count = count + ?", (new_rows_count,))
    conn.commit()
    
    # 2. Check threshold
    current_count = cursor.execute("SELECT count FROM stockpile").fetchone()[0]
    
    if current_count >= threshold:
        print(f"🚀 Threshold met ({current_count}). Triggering CT Pipeline...")
        
        # 3. Use Repository Dispatch to trigger GitHub Action
        response = requests.post(
            f"https://api.github.com/repos/{os.getenv('REPO_OWNER')}/{os.getenv('REPO_NAME')}/dispatches",
            headers={
                "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
                "Accept": "application/vnd.github.v3+json"
            },
            json={"event_type": "retrain_models"}
        )
        
        if response.status_code == 204:
            # Reset counter only if trigger was successful
            cursor.execute("UPDATE stockpile SET count = 0")
            conn.commit()
            
    conn.close()