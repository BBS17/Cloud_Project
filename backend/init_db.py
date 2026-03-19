# Purpose: create tables

import psycopg2
import os
from dotenv import load_dotenv

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

        # create database if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created")
        else:
            print(f"Database '{DB_NAME}' already exists")

        cursor.close()
        conn.close()

    except Exception as e:
        print("Error creating database:", e)


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

        print("Table created successfully")

    except Exception as e:
        print("Error creating table:", e)


if __name__ == "__main__":
    create_database()
    create_table()