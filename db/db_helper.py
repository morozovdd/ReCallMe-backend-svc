import os
import psycopg2
import numpy as np
from typing import List, Dict

class FaceDatabase:
    def __init__(self):
        # Database connection parameters (replace with your Google Cloud SQL details)
        self.connection_params = {
            "host": os.getenv("DB_HOST", "34.42.252.84"),
            "database": os.getenv("DB_NAME", "cloud-sql-recallme"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }
        
        # Initialize database on first run
        self.setup_database()
    
    def setup_database(self):
        """Set up database tables using SQL file"""
        try:
            with psycopg2.connect(**self.connection_params) as conn:
                with conn.cursor() as cur:
                    # Read and execute SQL file
                    sql_path = os.path.join(
                        os.path.dirname(__file__), 
                        'migrations', 
                        'V1_create_tables.sql'
                    )
                    with open(sql_path, 'r') as sql_file:
                        cur.execute(sql_file.read())
                    conn.commit()
                    print("Database setup completed successfully!")
        except Exception as e:
            print(f"Error setting up database: {e}")
    
    def add_patient(self, first_name: str, last_name: str) -> int:
        """Add a new patient"""
        with psycopg2.connect(**self.connection_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO patients (first_name, last_name)
                    VALUES (%s, %s)
                    RETURNING patient_id
                """, (first_name, last_name))
                patient_id = cur.fetchone()[0]
                conn.commit()
                return patient_id

    def add_person(self, 
                  patient_id: int, 
                  first_name: str,
                  last_name: str, 
                  relationship: str, 
                  face_embedding: np.ndarray) -> int:
        """Add a new person to the database"""
        with psycopg2.connect(**self.connection_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO people 
                        (patient_id, first_name, last_name, 
                         relationship, face_embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (patient_id, first_name, last_name, 
                      relationship, face_embedding.tolist()))
                person_id = cur.fetchone()[0]
                conn.commit()
                return person_id

    def find_similar_face(self, 
                         face_embedding: np.ndarray, 
                         threshold: float = 0.8) -> List[Dict]:
        """Find similar faces in database"""
        with psycopg2.connect(**self.connection_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        p.id,
                        p.first_name,
                        p.last_name,
                        p.relationship,
                        1 - (p.face_embedding <=> %s) as similarity
                    FROM people p
                    WHERE 1 - (p.face_embedding <=> %s) > %s
                    ORDER BY similarity DESC
                    LIMIT 5
                """, (face_embedding.tolist(), 
                      face_embedding.tolist(), 
                      threshold))
                
                return [{
                    'id': row[0],
                    'first_name': row[1],
                    'last_name': row[2],
                    'relationship': row[3],
                    'similarity': float(row[4])
                } for row in cur.fetchall()]