"""
Database connection and initialization module using asyncpg.
"""
import os
import asyncpg
import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# PostgreSQL connection parameters from environment variables
PG_HOST = os.environ.get('MBZUAI_PG_HOST', 'localhost')
PG_PORT = os.environ.get('MBZUAI_PG_PORT', '5432')
PG_USER = os.environ.get('MBZUAI_PG_USER', 'postgres')
PG_PASSWORD = os.environ.get('MBZUAI_PG_PASSWORD', 'postgres')
PG_DATABASE = os.environ.get('MBZUAI_PG_DATABASE', 'mbzuai_database')

# Construct DSN (Data Source Name)
DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# Log database connection info (without password)
logger.info(f"Configuring asyncpg pool for database: {PG_DATABASE} on {PG_HOST}:{PG_PORT}")

async def connect_db() -> Optional[asyncpg.Pool]:
    """Create and return the asyncpg connection pool."""
    logger.info("Attempting to create asyncpg connection pool...")
    pool = None # Initialize pool variable for this function scope
    try:
        pool = await asyncpg.create_pool(
            dsn=DSN,
            min_size=5,   # Minimum number of connections in the pool
            max_size=20   # Maximum number of connections in the pool
        )
        logger.info("Asyncpg connection pool created successfully.")
        if pool:
            logger.info(f"Pool object created: {pool}")
        else:
            logger.warning("Pool object is None IMMEDIATELY after creation call.")
    except Exception as e:
        logger.error(f"Failed to create asyncpg connection pool: {e}")
        pool = None # Ensure pool is None if connection failed
    logger.info(f"connect_db finished. Returning pool: {pool}")
    return pool

async def disconnect_db(pool: Optional[asyncpg.Pool]):
    """Close the asyncpg connection pool during application shutdown."""
    if pool:
        logger.info(f"Attempting to close connection pool: {pool}")
        await pool.close()
        logger.info("Asyncpg connection pool closed.")
    else:
        logger.warning("disconnect_db called but pool was None or not provided.")

async def init_db(pool: Optional[asyncpg.Pool]) -> None:
    """
    Initialize the PostgreSQL database asynchronously using asyncpg.
    Creates tables and indices if they don't exist.
    """
    # Reminder about backups
    logger.info("PostgreSQL backups should be configured at the database server level.")
    logger.info("Consider using pg_dump, automated backup solutions, or consulting your DBA.")
    
    if not pool:
        logger.error("Database pool is not initialized. Cannot run init_db.")
        return

    async with pool.acquire() as conn:
        async with conn.transaction():
            try:
                # Create chat_history table
                await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    feedback TEXT CHECK(feedback IN ('like', 'dislike', NULL)),
                    id_str TEXT UNIQUE,
                    rag_app_name TEXT
                )
                ''')
                
                # Create sources table
                await conn.execute('''
                CREATE TABLE IF NOT EXISTS sources (
                    id SERIAL PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    cite_num TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chat_history (id) ON DELETE CASCADE
                )
                ''')
                
                # Create indices
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history(timestamp)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_sources_chat_id ON sources(chat_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_id_str ON chat_history(id_str)') # Index for id_str

                # Schema versioning and migration
                await conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                ''')

                # Check current schema version
                result = await conn.fetchrow('SELECT MAX(version) as max_version FROM schema_version')
                current_version = result['max_version'] if result and result['max_version'] is not None else 0

                # Migration: Add rag_app_name column if schema version < 3
                if current_version < 3:
                    try:
                        # Check if column exists
                        column_exists = await conn.fetchrow("""
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='chat_history' AND column_name='rag_app_name'
                        """)
                        if not column_exists:
                            await conn.execute('ALTER TABLE chat_history ADD COLUMN rag_app_name TEXT')
                            logger.info("Added rag_app_name column to chat_history table.")
                        
                        # Update schema version
                        await conn.execute('INSERT INTO schema_version (version) VALUES ($1)', 3)
                        logger.info("Applied schema migration to version 3")
                    except Exception as mig_err:
                        logger.error(f"Error during migration to version 3: {mig_err}")
                        raise

                # Migration: Add id_str column if schema version < 2
                if current_version < 2:
                    try:
                        # Check if column exists (using information_schema)
                        column_exists = await conn.fetchrow("""
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name='chat_history' AND column_name='id_str'
                        """)
                        if not column_exists:
                            await conn.execute('ALTER TABLE chat_history ADD COLUMN id_str TEXT UNIQUE')
                            logger.info("Added id_str column to chat_history table.")
                        
                        # Update schema version within the same transaction
                        await conn.execute('INSERT INTO schema_version (version) VALUES ($1)', 2)
                        logger.info("Applied schema migration to version 2")
                    except Exception as mig_err:
                        logger.error(f"Error during migration to version 2: {mig_err}")
                        # Transaction will be rolled back automatically
                        raise # Re-raise to indicate failure

                logger.info("Database initialized successfully (or already up-to-date).")

            except asyncpg.PostgresError as e:
                logger.error(f"Database initialization error: {e}")
                # Transaction automatically rolled back
            except Exception as e:
                logger.error(f"Unexpected error during database initialization: {e}")
                # Transaction automatically rolled back
