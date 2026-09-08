import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def run_migrations():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db = os.getenv("POSTGRES_DB", "chakra")

    print(f"Connecting to postgres://{user}:***@{host}:{port}/{db}")
    conn = await asyncpg.connect(
        user=user, password=password, database=db, host=host, port=port
    )
    
    try:
        # Create schema_migrations table if not exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        migrations_dir = os.path.join(os.path.dirname(__file__), "..", "app", "db", "migrations")
        files = sorted([f for f in os.listdir(migrations_dir) if f.endswith(".sql")])
        
        for file in files:
            version = file.split("_")[0]
            
            # Check if applied
            row = await conn.fetchrow("SELECT version FROM schema_migrations WHERE version = $1", version)
            if not row:
                print(f"Applying migration {file}...")
                with open(os.path.join(migrations_dir, file), "r") as f:
                    sql = f.read()
                
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                print(f"Migration {file} applied successfully.")
            else:
                print(f"Migration {file} already applied.")
                
    finally:
        await conn.close()
        print("Migrations complete.")

if __name__ == "__main__":
    asyncio.run(run_migrations())
