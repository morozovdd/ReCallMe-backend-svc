import os
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    # For Google Cloud SQL, the host will be the Public IP address
    host: str = os.getenv('DB_HOST', 'p1031256824093-h860kl@gcp-sa-cloud-sql.iam.gserviceaccount.com')  # Replace with your instance's IP
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'cloud-sql-recallme')
    user: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', '1Y74*0)_f0I&s|0a')  # Never hardcode password here

# Test configuration
def test_connection(config: DatabaseConfig):
    import psycopg2
    print("Attempting to connect to the database...")
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"Database: {config.database}")
    print(f"User: {config.user}")
    
    try:
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            connect_timeout=10  
        )
        conn.close()
        print("Database connection successful!")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test the connection
    config = DatabaseConfig()
    test_connection(config)