# Purpose: create tables

import logging
import os

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DB_NAME = os.getenv("DB_NAME")

def create_database():
    try:
        # connect to default postgres database
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database="postgres",  # IMPORTANT
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )

        conn.autocommit = True  # REQUIRED for CREATE DATABASE
        cursor = conn.cursor()

        # Use a parameterized query to safely check existence
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()

        if not exists:
            # Use sql.Identifier to safely quote the database name
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            logger.info("Database '%s' created", DB_NAME)
        else:
            logger.info("Database '%s' already exists", DB_NAME)

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error("Error creating database: %s", e)


def create_table():
    try:
        # now connect to your actual database
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=DB_NAME,
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_logs (
                id SERIAL PRIMARY KEY,
                input_length INT,
                prediction TEXT,
                latency_ms FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("Table created successfully")

    except Exception as e:
        logger.error("Error creating table: %s", e)


if __name__ == "__main__":
    create_database()
    create_table()