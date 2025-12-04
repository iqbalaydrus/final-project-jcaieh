import sqlite3
import json
import pandas as pd

# Define the path to your data and the name for your database
JSONL_FILE_PATH = 'data/indonesian_job_postings.jsonl' #<-- Make sure this path is correct
DB_PATH = 'database.db'
TABLE_NAME = 'jobs'

def create_database():
    # These are the columns you want in your SQL database
    # Adjust them if needed based on the dataset
    sql_columns = [
        'work_type', 'salary', 'location', 'company_name',
        'job_title', '_scrape_timestamp'
    ]

    # Read the jsonl data
    records = []
    with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Keep only the columns you need for the SQL table
    df_sql = df[sql_columns]

    # Connect to SQLite and save the data
    try:
        conn = sqlite3.connect(DB_PATH)
        df_sql.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        print(f"Database '{DB_PATH}' created successfully with table '{TABLE_NAME}'.")
        
        # Verify by fetching a few rows
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
        rows = cursor.fetchall()
        print("\nFirst 5 rows from the new database table:")
        for row in rows:
            print(row)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    create_database()
