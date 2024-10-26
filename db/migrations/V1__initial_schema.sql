-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables for RecallMe face recognition system
CREATE TABLE IF NOT EXISTS patients (
    patient_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS people (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    relationship VARCHAR(50) NOT NULL,
    face_embedding vector(512),  -- Adjust size based on your model's output
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);