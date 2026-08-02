import logging
import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),
    )


@contextmanager
def _db_cursor():
    """Context manager that opens a connection, yields a cursor, commits, and always closes."""
    conn = _get_connection()
    try:
        with conn.cursor() as cursor:
            yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_db() -> None:
    """Raises if the database is not reachable. Used by the health endpoint."""
    with _db_cursor() as cursor:
        cursor.execute("SELECT 1")


def log_inference(text: str, response: dict, latency_ms: float) -> None:
    try:
        with _db_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO inference_logs (input_length, prediction, latency_ms)
                VALUES (%s, %s, %s)
                """,
                (len(text), response.get("label"), latency_ms),
            )
        logger.info("Inference log saved (latency=%.1f ms)", latency_ms)
    except Exception as exc:
        logger.error("Failed to log inference: %s", exc)


def get_metrics_summary() -> dict | None:
    try:
        with _db_cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    AVG(latency_ms),
                    MIN(latency_ms),
                    MAX(latency_ms)
                FROM inference_logs
                """
            )
            result = cursor.fetchone()

        return {
            "total_requests": result[0],
            "avg_latency_ms": round(float(result[1]), 2) if result[1] else 0.0,
            "min_latency_ms": round(float(result[2]), 2) if result[2] else 0.0,
            "max_latency_ms": round(float(result[3]), 2) if result[3] else 0.0,
        }
    except Exception as exc:
        logger.error("Failed to fetch metrics: %s", exc)
        return None
