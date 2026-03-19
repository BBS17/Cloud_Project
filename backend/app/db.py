import psycopg2
import os
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Connection helper
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )

# Insert new log in database
def log_inference(text, response, latency_ms):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO inference_logs (input_length, prediction, latency_ms)
            VALUES (%s, %s, %s)
            """,
            (
                len(text),
                str(response),
                latency_ms
            )
        )

        conn.commit()

        cursor.close()
        conn.close()

        print("Inserted inference log successfully")

    except Exception as e:
        print("Database logging error:", e)

# Return metrics data
def get_metrics_summary():
    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*),
                    AVG(latency_ms),
                    MIN(latency_ms),
                    MAX(latency_ms)
                FROM inference_logs
            """)

            result = cursor.fetchone()

        conn.close()

        return {
            "total_requests": result[0],
            "avg_latency": float(result[1]) if result[1] else 0,
            "min_latency": float(result[2]) if result[2] else 0,
            "max_latency": float(result[3]) if result[3] else 0
        }

    except Exception as e:
        print("Metrics error:", e)
        return {}